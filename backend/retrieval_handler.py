import os
import json
import asyncio
import re # Import the regular expression module
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import CrossEncoder

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash-latest"
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "document_insights"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'

# --- Prompts ---

# *** NEW: Specialized prompts for parallel execution ***
CONTRADICTION_PROMPT = """
You are a highly intelligent AI assistant specializing in document analysis. Your task is to analyze a user's selected text and a large list of context sections to find **only the contradictions**.
1.  Analyze all the provided context sections.
2.  Identify the top 5 most relevant sections that present a viewpoint or fact that directly opposes or challenges the user's selected text.
3.  Return a JSON object with a single key "contradictions" containing an array of the full JSON objects for the sections you have selected.
If you find no contradictions, return an empty array. Do not add any explanatory text.

USER'S SELECTED TEXT:
"{user_selection}"

LARGE CONTEXT:
{context_sections_json}
---
"""

ENHANCEMENT_PROMPT = """
You are a highly intelligent AI assistant specializing in document analysis. Your task is to analyze a user's selected text and a large list of context sections to find **only the enhancements**.
1.  Analyze all the provided context sections.
2.  Identify the top 5 most relevant sections that provide a more detailed explanation, a specific example, or build directly upon the user's selection.
3.  Return a JSON object with a single key "enhancements" containing an array of the full JSON objects for the sections you have selected.
If you find no enhancements, return an empty array. Do not add any explanatory text.

USER'S SELECTED TEXT:
"{user_selection}"

LARGE CONTEXT:
{context_sections_json}
---
"""

CONNECTION_PROMPT = """
You are a highly intelligent AI assistant specializing in document analysis. Your task is to analyze a user's selected text and a large list of context sections to find **only the connections**.
1.  Analyze all the provided context sections.
2.  Identify the top 5 most relevant sections that are thematically related to the user's selection but are not direct enhancements or contradictions.
3.  Return a JSON object with a single key "connections" containing an array of the full JSON objects for the sections you have selected.
If you find no connections, return an empty array. Do not add any explanatory text.

USER'S SELECTED TEXT:
"{user_selection}"

LARGE CONTEXT:
{context_sections_json}
---
"""

# *** UPDATED: New podcast prompt for conversation array format ***
PERSONA_PODCAST_PROMPT = """
You are a creative podcast script writer. You will be given a user's selected text and a large list of up to 200 context sections.
Your task is to analyze all the sections to find the most interesting and relevant information, then create a conversational podcast in the style of a "{persona}".

Generate a natural conversation between a Host and an Analyst following the {persona} style. The conversation should be 8-12 exchanges total (4-6 from each speaker).

Return ONLY a JSON object with a single key "conversation" containing an array of strings where:
- Even indices (0, 2, 4, ...): Host speaking
- Odd indices (1, 3, 5, ...): Analyst speaking

Each exchange should be 1-3 sentences and sound natural and conversational. Do not include speaker names in the text - just the dialogue.

Style Guide: {style_guide}

USER'S SELECTED TEXT:
"{user_selection}"

LARGE CONTEXT (Up to 200 sections):
{context_sections_json}

Example format:
{{"conversation": ["Welcome to our podcast! Today we're exploring...", "Thanks! This topic is fascinating because...", "That's a great point. What I find interesting is...", "Exactly, and the data shows..."]}}
"""

PERSONA_STYLES = {
    "debater": "The Host and Analyst should present opposing viewpoints or debate the nuances of the findings.",
    "investigator": "The Host and Analyst should dig deep into the evidence, questioning assumptions and focusing on factual details.",
    "fundamentals": "The Host and Analyst should start with the most basic, foundational concepts from the context and progressively build up to the user's selected topic.",
    "connections": "The Host and Analyst should focus on drawing surprising connections and analogies between the selected topic and other concepts found in the context, even from different domains."
}

def extract_json_from_string(text: str) -> Dict[str, Any]:
    """
    Basic JSON extraction function - kept for backward compatibility.
    Use specialized functions for specific endpoint requirements.
    """
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        print("Error: No JSON object found in the model's response string.")
        return {"error": "No JSON object found in response."}
    json_str = json_match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic JSON string found by regex: '{json_str}'")
        return {"error": "Failed to parse JSON from model response."}
    except Exception as e:
        print(f"An unexpected error occurred during JSON extraction: {e}")
        return {"error": "An unexpected error occurred."}

def extract_insights_from_response(text: str, expected_key: str, insight_type: str) -> List[Dict[str, Any]]:
    """
    Robust extraction function specifically for insights (contradictions, enhancements, connections).
    Handles LLM hallucinations and various malformed outputs.
    """
    print(f"Extracting {insight_type} insights from response...")
    
    # Try JSON extraction first
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            
            # Check for expected key
            if expected_key in parsed_json:
                insights = parsed_json[expected_key]
                if isinstance(insights, list):
                    # Validate each insight is a dictionary
                    validated_insights = []
                    for insight in insights:
                        if isinstance(insight, dict):
                            # Ensure required fields exist, add defaults if missing
                            if 'original_content' not in insight:
                                insight['original_content'] = f"Content not available for this {insight_type}"
                            if 'page_number' not in insight:
                                insight['page_number'] = 0
                            if 'document_name' not in insight:
                                insight['document_name'] = "Unknown Document"
                            validated_insights.append(insight)
                        elif isinstance(insight, str):
                            # Convert string to proper insight format
                            validated_insights.append({
                                'original_content': insight,
                                'page_number': 0,
                                'document_name': 'Unknown Document',
                                'bounding_box': {}
                            })
                    
                    if validated_insights:
                        print(f"Successfully extracted {len(validated_insights)} {insight_type} insights")
                        return validated_insights
                else:
                    print(f"{expected_key} is not a list, checking for alternative formats")
            
            # Check for alternative key names (common hallucinations)
            alternative_keys = {
                'contradictions': ['contradictory', 'opposing', 'conflicts', 'disagreements'],
                'enhancements': ['details', 'expansions', 'elaborations', 'specifics'],
                'connections': ['related', 'links', 'associations', 'relationships']
            }
            
            for alt_key in alternative_keys.get(expected_key, []):
                if alt_key in parsed_json:
                    print(f"Found alternative key '{alt_key}' for {insight_type}")
                    alt_insights = parsed_json[alt_key]
                    if isinstance(alt_insights, list) and alt_insights:
                        return extract_insights_from_response(json.dumps({expected_key: alt_insights}), expected_key, insight_type)
            
            # Check if response is wrapped in unexpected structure
            for key, value in parsed_json.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"Found potential insights in key '{key}', attempting to use")
                    return extract_insights_from_response(json.dumps({expected_key: value}), expected_key, insight_type)
                    
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {insight_type}: {e}")
    
    # Fallback: Return empty list for clean handling
    print(f"No valid {insight_type} found, returning empty list")
    return []

def extract_podcast_conversation_from_response(text: str, persona: str) -> List[str]:
    """
    Specialized function to extract and validate podcast conversation arrays from LLM responses.
    Handles various hallucination scenarios and malformed outputs.
    """
    print(f"Extracting podcast conversation for persona: {persona}")
    
    # First, try to extract JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            
            # Check if it has the expected "conversation" key
            if "conversation" in parsed_json:
                conversation = parsed_json["conversation"]
                
                # Validate that conversation is a list
                if isinstance(conversation, list):
                    # Validate that all items are strings
                    if all(isinstance(item, str) for item in conversation):
                        # Ensure we have at least 4 exchanges and even number for proper alternation
                        if len(conversation) >= 4:
                            # Ensure even number of exchanges for proper Host/Analyst alternation
                            if len(conversation) % 2 == 0:
                                print(f"Successfully extracted {len(conversation)} conversation exchanges")
                                return conversation
                            else:
                                # Odd number - remove last item to make it even
                                print(f"Odd number of exchanges ({len(conversation)}), trimming to even")
                                return conversation[:-1]
                        else:
                            print(f"Too few exchanges ({len(conversation)}), using fallback")
                    else:
                        print("Conversation contains non-string items, using fallback")
                else:
                    print("Conversation is not a list, using fallback")
            else:
                # Check for alternative key names (common hallucinations)
                alternative_keys = ["dialogue", "script", "podcast", "messages", "exchanges", "lines"]
                for key in alternative_keys:
                    if key in parsed_json:
                        print(f"Found alternative key '{key}', attempting to use it")
                        alt_conversation = parsed_json[key]
                        if isinstance(alt_conversation, list) and len(alt_conversation) >= 4:
                            if all(isinstance(item, str) for item in alt_conversation):
                                # Ensure even number
                                if len(alt_conversation) % 2 != 0:
                                    alt_conversation = alt_conversation[:-1]
                                return alt_conversation
                        break
                print("No valid conversation key found in JSON, using fallback")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, attempting text parsing fallback")
    
    # Fallback 1: Try to extract dialogue from plain text using patterns
    print("Attempting to parse conversation from plain text...")
    
    # Look for common dialogue patterns
    dialogue_patterns = [
        r'"([^"]+)"',  # Text in quotes
        r'Host[:\s]+([^\n]+)',  # Lines starting with Host:
        r'Analyst[:\s]+([^\n]+)',  # Lines starting with Analyst:
        r'•\s*([^\n]+)',  # Bullet points
        r'\d+\.\s*([^\n]+)',  # Numbered lines
        r'-\s*([^\n]+)',  # Dashed lines
    ]
    
    extracted_lines = []
    for pattern in dialogue_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Clean and filter matches
            cleaned_matches = [match.strip() for match in matches if len(match.strip()) > 10]
            if len(cleaned_matches) >= 4:
                extracted_lines = cleaned_matches[:12]  # Limit to 12 exchanges max
                break
    
    if extracted_lines and len(extracted_lines) >= 4:
        # Ensure even number
        if len(extracted_lines) % 2 != 0:
            extracted_lines = extracted_lines[:-1]
        print(f"Successfully extracted {len(extracted_lines)} lines from text patterns")
        return extracted_lines
    
    # Fallback 2: Generate a safe default conversation
    print(f"All parsing attempts failed, generating safe fallback for persona: {persona}")
    
    fallback_conversations = {
        "debater": [
            "Welcome to our debate-style analysis of your selected text.",
            "I believe this topic has some controversial aspects worth discussing.",
            "That's an interesting perspective. However, I see some counterarguments.",
            "Fair point, but let's consider the evidence from multiple angles.",
            "The data seems to support both viewpoints to some degree.",
            "True, this complexity makes it a fascinating subject for analysis.",
            "Let's explore what the context reveals about these different positions.",
            "The insights suggest this topic deserves deeper investigation."
        ],
        "investigator": [
            "Let's investigate your selected text with a critical eye.",
            "The evidence in our context provides several key insights.",
            "What specific details support these findings?",
            "The documentation shows clear patterns worth examining.",
            "Are there any gaps in the information we should note?",
            "Good question - some areas definitely need more investigation.",
            "The factual analysis reveals important connections.",
            "This investigation approach helps uncover hidden insights."
        ],
        "fundamentals": [
            "Let's start with the basic concepts underlying your selected text.",
            "Understanding the fundamentals is crucial for deeper analysis.",
            "What are the core principles we should establish first?",
            "The foundational elements include several key components.",
            "How do these basics connect to more advanced concepts?",
            "That's where it gets interesting - the progression is quite logical.",
            "Building from these fundamentals leads to richer understanding.",
            "Exactly, this step-by-step approach reveals the full picture."
        ],
        "connections": [
            "Let's explore the surprising connections in your selected text.",
            "This topic links to several unexpected areas worth discussing.",
            "What patterns do you see emerging across different domains?",
            "The connections span multiple fields in fascinating ways.",
            "Are there analogies that might help illustrate these relationships?",
            "Great question - there are some compelling parallels to consider.",
            "These cross-domain insights reveal deeper underlying principles.",
            "The interconnected nature of knowledge always amazes me."
        ]
    }
    
    return fallback_conversations.get(persona, [
        "Welcome to our podcast discussion.",
        "Unfortunately, we encountered some technical difficulties.",
        "Let's explore your selected text as best we can.",
        "The context provides valuable insights worth discussing.",
        "Thank you for your patience with this analysis.",
        "We hope this perspective is still helpful.",
        "Please feel free to try again for better results.",
        "Thanks for listening to our discussion."
    ])


class RetrievalHandler:
    def __init__(self, google_api_key: str):
        if not google_api_key:
            raise ValueError("Google API key is required.")
        genai.configure(api_key=google_api_key)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.generation_model = genai.GenerativeModel(
            GENERATION_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print("RetrievalHandler initialized successfully with reranker.")

    async def retrieve_fast_async(self, user_selection: str) -> List[Dict[str, Any]]:
        """
        Fast retrieval with comprehensive error handling and validation.
        """
        print("Executing FAST retrieval (single-query + rerank)...")
        
        try:
            # Validate input
            if not user_selection or len(user_selection.strip()) < 3:
                print("User selection too short or empty")
                return []
            
            # Generate embedding
            result = await genai.embed_content_async(
                model=EMBEDDING_MODEL, 
                content=user_selection, 
                task_type="RETRIEVAL_QUERY"
            )
            
            if not result or 'embedding' not in result:
                print("Failed to generate embedding for user selection")
                return []
                
            query_embedding = result['embedding']
            
            # Query the collection
            query_results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=100
            )
            
            candidate_metadatas = query_results.get('metadatas', [[]])[0]
            if not candidate_metadatas: 
                print("No results found in vector database")
                return []
            
            # Rerank results
            try:
                rerank_pairs = [[user_selection, meta.get('original_content', '')] for meta in candidate_metadatas]
                scores = self.reranker.predict(rerank_pairs)
                scored_candidates = sorted(zip(scores, candidate_metadatas), key=lambda x: x[0], reverse=True)
                reranked_results = [meta for score, meta in scored_candidates[:30]]
            except Exception as rerank_error:
                print(f"Reranking failed, using original order: {rerank_error}")
                reranked_results = candidate_metadatas[:30]
            
            # Validate and clean results
            unique_results = []
            seen_content = set()
            for meta in reranked_results:
                try:
                    # Validate metadata structure
                    if not isinstance(meta, dict):
                        continue
                        
                    content = meta.get('original_content', '')
                    if not content or content in seen_content:
                        continue
                        
                    seen_content.add(content)
                    
                    # Ensure required fields exist
                    if 'page_number' not in meta:
                        meta['page_number'] = 0
                    if 'document_name' not in meta:
                        meta['document_name'] = 'Unknown Document'
                    
                    # Handle bounding box
                    if 'bounding_box' in meta and isinstance(meta['bounding_box'], str):
                        try: 
                            meta['bounding_box'] = json.loads(meta['bounding_box'])
                        except json.JSONDecodeError: 
                            meta['bounding_box'] = {}
                    elif 'bounding_box' not in meta:
                        meta['bounding_box'] = {}
                        
                    unique_results.append(meta)
                    
                except Exception as meta_error:
                    print(f"Error processing metadata: {meta_error}")
                    continue

            print(f"Fast retrieval complete. Found {len(unique_results)} unique sections.")
            return unique_results
            
        except Exception as e:
            print(f"Error in fast retrieval: {type(e).__name__} - {e}")
            return []

    async def retrieve_large_context_async(self, user_selection: str) -> List[Dict[str, Any]]:
        """
        Large context retrieval with comprehensive error handling and validation.
        """
        print("Executing LARGE context retrieval (retrieving up to 200 sections)...")
        
        try:
            # Validate input
            if not user_selection or len(user_selection.strip()) < 3:
                print("User selection too short or empty")
                return []
                
            # Generate embedding
            result = await genai.embed_content_async(
                model=EMBEDDING_MODEL, 
                content=user_selection, 
                task_type="RETRIEVAL_QUERY"
            )
            
            if not result or 'embedding' not in result:
                print("Failed to generate embedding for user selection")
                return []
                
            query_embedding = result['embedding']
            
            # Query the collection
            query_results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=200
            )
            
            candidate_metadatas = query_results.get('metadatas', [[]])[0]
            if not candidate_metadatas:
                print("No results found in vector database")
                return []
            
            # Validate and clean results
            unique_results = []
            seen_content = set()
            processed_count = 0
            error_count = 0
            
            for meta in candidate_metadatas:
                try:
                    processed_count += 1
                    
                    # Validate metadata structure
                    if not isinstance(meta, dict):
                        error_count += 1
                        continue
                        
                    content = meta.get('original_content', '')
                    if not content or content in seen_content:
                        continue
                        
                    seen_content.add(content)
                    
                    # Ensure required fields exist with defaults
                    if 'page_number' not in meta:
                        meta['page_number'] = 0
                    if 'document_name' not in meta:
                        meta['document_name'] = 'Unknown Document'
                    
                    # Handle bounding box parsing
                    if 'bounding_box' in meta and isinstance(meta['bounding_box'], str):
                        try:
                            meta['bounding_box'] = json.loads(meta['bounding_box'])
                        except json.JSONDecodeError:
                            meta['bounding_box'] = {}
                    elif 'bounding_box' not in meta:
                        meta['bounding_box'] = {}
                    
                    unique_results.append(meta)
                    
                except Exception as meta_error:
                    error_count += 1
                    print(f"Error processing metadata item {processed_count}: {meta_error}")
                    continue
            
            print(f"Large context retrieval complete. Processed: {processed_count}, Errors: {error_count}, Unique results: {len(unique_results)}")
            return unique_results

        except Exception as e:
            print(f"Error in large context retrieval: {type(e).__name__} - {e}")
            return []

    async def find_enhancements_async(self, user_selection: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = ENHANCEMENT_PROMPT.format(user_selection=user_selection, context_sections_json=json.dumps(context))
        try:
            response = await self.generation_model.generate_content_async(prompt)
            if not response.parts: 
                print("Enhancement generation was blocked")
                return []
            
            # Use robust extraction function
            return extract_insights_from_response(response.text, "enhancements", "enhancement")
            
        except Exception as e:
            print(f"Error finding enhancements: {type(e).__name__} - {e}")
            return []

    async def find_connections_async(self, user_selection: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = CONNECTION_PROMPT.format(user_selection=user_selection, context_sections_json=json.dumps(context))
        try:
            response = await self.generation_model.generate_content_async(prompt)
            if not response.parts:
                print("Connection generation was blocked")
                return []
            
            # Use robust extraction function
            return extract_insights_from_response(response.text, "connections", "connection")
            
        except Exception as e:
            print(f"Error finding connections: {type(e).__name__} - {e}")
            return []

    async def find_contradictions_async(self, user_selection: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = CONTRADICTION_PROMPT.format(user_selection=user_selection, context_sections_json=json.dumps(context))
        try:
            response = await self.generation_model.generate_content_async(prompt)
            if not response.parts:
                print("Contradiction generation was blocked")
                return []
            
            # Use robust extraction function
            return extract_insights_from_response(response.text, "contradictions", "contradiction")
            
        except Exception as e:
            print(f"Error finding contradictions: {type(e).__name__} - {e}")
            return []

    # *** OPTIMIZED: Much faster parallel insights generation ***
    async def generate_initial_insights_async(self, user_selection: str) -> Dict[str, Any]:
        """
        Ultra-fast parallel insights generation using optimized context retrieval and concurrent processing.
        """
        print("Starting optimized insights generation with parallel processing...")
        
        # Get context once and reuse for all three insight types
        large_context = await self.retrieve_large_context_async(user_selection)
        if not large_context:
            print("No context found, returning empty insights")
            return {"contradictions": [], "enhancements": [], "connections": []}
        
        print(f"Retrieved {len(large_context)} context sections, generating all insights in parallel...")
        
        # Create all three tasks to run completely in parallel
        enhancements_task = asyncio.create_task(
            self.find_enhancements_async(user_selection, large_context),
            name="enhancements"
        )
        connections_task = asyncio.create_task(
            self.find_connections_async(user_selection, large_context),
            name="connections"
        )
        contradictions_task = asyncio.create_task(
            self.find_contradictions_async(user_selection, large_context),
            name="contradictions"
        )
        
        # Wait for all tasks to complete simultaneously
        try:
            results = await asyncio.gather(
                enhancements_task,
                connections_task,
                contradictions_task,
                return_exceptions=True  # Don't fail if one task fails
            )
            
            # Process results with error handling
            enhancements = results[0] if not isinstance(results[0], Exception) else []
            connections = results[1] if not isinstance(results[1], Exception) else []
            contradictions = results[2] if not isinstance(results[2], Exception) else []
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_names = ["enhancements", "connections", "contradictions"]
                    print(f"Task {task_names[i]} failed with exception: {result}")
            
            final_insights = {
                "enhancements": enhancements,
                "connections": connections,
                "contradictions": contradictions
            }
            
            total_insights = len(enhancements) + len(connections) + len(contradictions)
            print(f"Insights generation complete! Total insights: {total_insights}")
            print(f"  - Enhancements: {len(enhancements)}")
            print(f"  - Connections: {len(connections)}")
            print(f"  - Contradictions: {len(contradictions)}")
            
            return final_insights
            
        except Exception as e:
            print(f"Error in parallel insights generation: {type(e).__name__} - {e}")
            return {"contradictions": [], "enhancements": [], "connections": []}

    # *** UPDATED: Helper function with robust error handling ***
    async def _generate_single_podcast_script(self, selection: str, persona: str, style_guide: str, context: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generates a conversation array for a single persona with comprehensive error handling.
        """
        print(f"Generating podcast conversation for persona: {persona}...")
        prompt = PERSONA_PODCAST_PROMPT.format(
            persona=persona,
            style_guide=style_guide, 
            user_selection=selection, 
            context_sections_json=json.dumps(context, indent=2)
        )
        
        try:
            response = await self.generation_model.generate_content_async(prompt)
            
            # Check if response was blocked
            if not response.parts:
                block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'Unknown')
                print(f"Podcast generation was blocked for persona '{persona}'. Reason: {block_reason}")
                return persona, extract_podcast_conversation_from_response("", persona)
            
            # Check if response text is empty or too short
            response_text = response.text.strip()
            if not response_text or len(response_text) < 50:
                print(f"Response too short or empty for persona '{persona}', using fallback")
                return persona, extract_podcast_conversation_from_response("", persona)
            
            # Use specialized podcast extraction function
            conversation = extract_podcast_conversation_from_response(response_text, persona)
            
            print(f"Successfully processed conversation for persona: {persona} ({len(conversation)} exchanges)")
            return persona, conversation
            
        except Exception as e:
            print(f"Exception during podcast generation for persona '{persona}': {type(e).__name__} - {e}")
            # Return safe fallback using the specialized function
            return persona, extract_podcast_conversation_from_response("", persona)

    # *** UPDATED: Main function now returns conversation arrays ***
    async def generate_persona_podcast_async(self, selection: str) -> Dict[str, Any]:
        """
        Generates podcast conversations for all personas in the format:
        {
            "debater": ["Host message", "Analyst message", "Host message", "Analyst message", ...],
            "investigator": ["Host message", "Analyst message", ...],
            ...
        }
        """
        context_sections = await self.retrieve_large_context_async(selection)
        if not context_sections:
            return {persona: ["Cannot generate a podcast without context.", "Please ensure your documents are properly indexed."] for persona in PERSONA_STYLES}
            
        # Create a list of tasks, one for each persona
        tasks = []
        for persona, style_guide in PERSONA_STYLES.items():
            task = self._generate_single_podcast_script(selection, persona, style_guide, context_sections)
            tasks.append(task)
        
        # Run all podcast generation tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Convert the list of (persona, conversation) tuples into a final dictionary
        all_podcasts = {persona: conversation for persona, conversation in results}
        
        return all_podcasts