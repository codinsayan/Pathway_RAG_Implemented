import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, FileText } from "lucide-react";
import LoadingText from "./LoadingText";

// Update Section interface to allow empty original_content and flexible fields
interface Section {
  bounding_box?: {
    x0: number;
    x1: number;
    y0: number;
    y1: number;
  };
  document_name?: string;
  full_path?: string;
  original_content?: string;
  page_number?: number;
  section_title?: string;
}

interface RightPanelProps {
  selectedText: string;
  onAudioFormatSelect: (
    format: "debater" | "investigator" | "fundamentals" | "connections"
  ) => void;
  activeAudioFormat:
    | "debater"
    | "investigator"
    | "fundamentals"
    | "connections"
    | null;
  onSectionCardClick?: (docId: string, searchTerm: string) => void;
}

const RightPanel = ({
  selectedText,
  onAudioFormatSelect,
  activeAudioFormat,
  onSectionCardClick,
}: RightPanelProps) => {
  const [activeTab, setActiveTab] = useState<"related" | "insights">("related");
  const [activeInsightTab, setActiveInsightTab] = useState<
    "contradictions" | "enhancements" | "connections"
  >("contradictions");
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(60);

  // Related sections and AI insights from API
  const [relatedSections, setRelatedSections] = useState<Section[]>([]);
  const [aiInsights, setAiInsights] = useState<{
    contradictions: Section[];
    enhancements: Section[];
    connections: Section[];
    podcast_script: { line: string; speaker: string }[];
  }>({
    contradictions: [],
    enhancements: [],
    connections: [],
    podcast_script: [],
  });
  const [isLoadingRelated, setIsLoadingRelated] = useState(false);
  const [isLoadingInsights, setIsLoadingInsights] = useState(false);

  // New state for podcast data and loading
  const [podcastData, setPodcastData] = useState<any>(null);
  const [isPodcastLoading, setIsPodcastLoading] = useState(false);

  // Audio-related state
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Track which persona is currently generating audio
  const [loadingPersona, setLoadingPersona] = useState<string | null>(null);

  // Add state for transcript
  const [showTranscript, setShowTranscript] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && audioRef.current) {
      interval = setInterval(() => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime);
          setDuration(audioRef.current.duration || 0);

          // Check if audio has ended
          if (audioRef.current.ended) {
            setIsPlaying(false);
          }
        }
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  useEffect(() => {
    if (!selectedText) {
      setRelatedSections([]);
      setAiInsights({
        contradictions: [],
        enhancements: [],
        connections: [],
        podcast_script: [],
      });
      setPodcastData(null); // Clear podcast data when no text is selected
      setAudioUrl(null); // Clear audio URL
      setIsPlaying(false); // Stop playing
      return;
    }

    setIsLoadingRelated(true);
    setIsLoadingInsights(false); // Don't start insights loading yet
    setPodcastData(null); // Clear previous podcast data
    setAudioUrl(null); // Clear previous audio
    setIsPlaying(false); // Stop playing

    // Fetch related sections first, then insights and podcast
    const fetchAll = async () => {
      try {
        // Always fetch related sections first
        const relatedRes = await fetch(
          "http://localhost:8000/get_retrieved_sections",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection: selectedText }),
          }
        );
        const relatedData = await relatedRes.json();
        setRelatedSections(relatedData.retrieved_sections || []);
      } catch (error) {
        console.error("Error retrieving sections:", error);
        setRelatedSections([]);
      } finally {
        setIsLoadingRelated(false);
      }

      // Now fetch insights and podcast in parallel
      setIsLoadingInsights(true);
      let insightsData = null;
      let podcastDataResp = null;
      try {
        const [insightsRes, podcastRes] = await Promise.all([
          fetch("http://localhost:8000/get_generated_insights", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection: selectedText }),
          }),
          fetch("http://localhost:8000/get_persona_podcast", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection: selectedText }),
          })
        ]);
        insightsData = await insightsRes.json();
        podcastDataResp = await podcastRes.json();
        setAiInsights({
          contradictions: insightsData.contradictions || [],
          enhancements: insightsData.enhancements || [],
          connections: insightsData.connections || [],
          podcast_script: insightsData.podcast_script || [],
        });
        setPodcastData(podcastDataResp);
      } catch (error) {
        console.error("Error retrieving insights or podcast:", error);
        setAiInsights({
          contradictions: [],
          enhancements: [],
          connections: [],
          podcast_script: [],
        });
        setPodcastData(null);
      } finally {
        setIsLoadingInsights(false);
      }
    };

    fetchAll();
  }, [selectedText]);

  // New effect to fetch podcast data when both related sections and insights are loaded
  useEffect(() => {
    const shouldFetchPodcast =
      selectedText && !isLoadingRelated && !isLoadingInsights;

    if (shouldFetchPodcast && !podcastData && !isPodcastLoading) {
      const fetchPodcastData = async () => {
        setIsPodcastLoading(true);
        try {
          const response = await fetch(
            "http://localhost:8000/get_persona_podcast",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ selection: selectedText }),
            }
          );
          const data = await response.json();
          setPodcastData(data);
          console.log("Podcast data fetched:", data);
        } catch (error) {
          console.error("Error fetching podcast data:", error);
          setPodcastData(null);
        } finally {
          setIsPodcastLoading(false);
        }
      };

      fetchPodcastData();
    }
  }, [
    selectedText,
    isLoadingRelated,
    isLoadingInsights,
    podcastData,
    isPodcastLoading,
  ]);

  // Use images for audio format buttons
  const audioFormats = [
    { id: "debater", label: "Debater", img: "/chat icon.png" },
    { id: "investigator", label: "Investigator", img: "/inverstigate icon.png" },
    { id: "fundamentals", label: "Fundamentals", img: "/gear icon.png" },
    { id: "connections", label: "Connections", img: "/building icon.png" }
  ];

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (value: number[]) => {
    if (audioRef.current) {
      audioRef.current.currentTime = value[0];
      setCurrentTime(value[0]);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getFormatLabel = (format: string) => {
    const formatData = audioFormats.find((f) => f.id === format);
    return formatData ? formatData.label : format;
  };

  // Updated podcast click handler to show loading only on clicked button
  const handlePodcastClick = async (persona: string) => {
    if (!podcastData || !podcastData[persona]) {
      console.log("No podcast data available for", persona);
      return;
    }
    setLoadingPersona(persona);
    setIsGeneratingAudio(true);
    setIsPlaying(false); // Stop any current playback
    try {
      const response = await fetch("http://localhost:8000/generate_podcast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(podcastData[persona]),
      });
      if (!response.ok) {
        throw new Error("Failed to generate podcast");
      }
      const data = await response.json();
      const fullAudioUrl = `http://localhost:8000/${data.audio_path}`;
      setAudioUrl(fullAudioUrl);
      onAudioFormatSelect(persona as any);
      if (audioRef.current) {
        audioRef.current.src = fullAudioUrl;
        audioRef.current.load();
        audioRef.current.onloadedmetadata = () => {
          if (audioRef.current) {
            setDuration(audioRef.current.duration);
            setCurrentTime(0);
          }
        };
        audioRef.current.onended = () => {
          setIsPlaying(false);
          setCurrentTime(0);
        };
      }
    } catch (error) {
      console.error("Error generating podcast:", error);
    } finally {
      setIsGeneratingAudio(false);
      setLoadingPersona(null);
    }
  };

  // Check if podcast buttons should be disabled
  const arePodcastButtonsDisabled =
    isLoadingRelated ||
    isLoadingInsights ||
    isPodcastLoading ||
    !podcastData ||
    isGeneratingAudio;

  // Update renderSectionList to handle empty original_content and show all fields
  const renderSectionList = (sections: Section[]) => (
    <div className="space-y-3">
      {sections.map((section, idx) => {
        const docId = section.document_name;
        return (
          <button
            key={idx}
            className="break-all w-full text-left p-3 bg-card border border-border rounded-lg hover:shadow-sm transition-all duration-200"
            onClick={() => {
              if (onSectionCardClick && docId) {
                const trimmedContent =
                  section.original_content &&
                  section.original_content.trim() !== ""
                    ? section.original_content.slice(0, 30)
                    : section.section_title || section.full_path || "";
                onSectionCardClick(docId, trimmedContent);
              }
            }}
            title={section.section_title}
          >
            <div className="break-all font-bold text-base mb-1">
              {section.section_title || section.full_path || "Untitled"}
            </div>
            {section.original_content &&
            section.original_content.trim() !== "" ? (
              <div className="text-xs text-foreground mb-2 line-clamp-3">
                {section.original_content.length > 120
                  ? section.original_content.slice(0, 120) + "..."
                  : section.original_content}
              </div>
            ) : null}
            <div className="text-xs text-muted-foreground mb-2 line-clamp-2">
              {section.full_path}
            </div>
            <div className="text-xs text-muted-foreground">
              Page: {section.page_number} | Doc: {section.document_name}
            </div>
            
            <div className="text-right mt-2">
              <span className="text-2xl">"</span>
            </div>
          </button>
        );
      })}
    </div>
  );

  // Loading bar component
  const LoadingBar = () => (
    <div className="w-full h-1 bg-muted/40 rounded overflow-hidden mb-4">
      <div className="h-full bg-primary animate-pulse" style={{ width: '60%' }}></div>
    </div>
  );

  // Loading spinner component
  const LoadingSpinner = () => (
    <span className="w-6 h-6 border-4 border-primary border-t-transparent rounded-full animate-spin mb-2 block mx-auto" />
  );

  // Loading spinner and animated loading text
  const relatedLoadingMessages = [
    "Finding best sections...",
    "Filtering out unwanted sections...",
    "Connecting the dots..."
  ];
  const insightsLoadingMessages = [
    "Making LLM call...",
    "Analysis by LLM...",
    "LLM crafting a response...",
    "Connecting the dots..."
  ];
  const [loadingMsgIdx, setLoadingMsgIdx] = useState(0);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (isLoadingRelated) {
      setLoadingMsgIdx(0);
      timer = setInterval(() => {
        setLoadingMsgIdx((prev) => (prev + 1) % relatedLoadingMessages.length);
      }, 3000);
    } else if (isLoadingInsights) {
      setLoadingMsgIdx(0);
      timer = setInterval(() => {
        setLoadingMsgIdx((prev) => (prev + 1) % insightsLoadingMessages.length);
      }, 3000);
    } else {
      setLoadingMsgIdx(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoadingRelated, isLoadingInsights]);

  return (
    <div className="h-full flex flex-col panel border-l">
      {/* Hidden audio element for playback */}
      <audio
        ref={audioRef}
        onLoadedMetadata={() => {
          if (audioRef.current) {
            setDuration(audioRef.current.duration);
          }
        }}
        onEnded={() => {
          setIsPlaying(false);
          setCurrentTime(0);
        }}
        onTimeUpdate={() => {
          if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
          }
        }}
      />

      {/* Selected Text */}
      <div className="py-4 pl-4 pr-1 border-b border-panel-border">
        <h3 className="font-medium text-sm mb-2">Selected Text</h3>
        <div
          className="text-sm text-muted-foreground italic max-h-24 overflow-y-auto break-words whitespace-pre-line custom-scrollbar"
          style={{ wordBreak: "break-word" }}
        >
          {selectedText || '"Select text from the document to see insights"'}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-panel-border">
        <button
          onClick={() => setActiveTab("related")}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
            activeTab === "related"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Related Sections
          {activeTab === "related" && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
          )}
        </button>
        <button
          onClick={() => setActiveTab("insights")}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
            activeTab === "insights"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          AI Insights
          {activeTab === "insights" && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
          )}
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto custom-scrollbar">
        {activeTab === "related" ? (
          <div className="p-4">
            {!selectedText ? (
              <div className="text-muted-foreground italic">
                Select text from the document to see related sections.
              </div>
            ) : isLoadingRelated ? (
              <div className="flex flex-col items-center justify-center mb-4">
                <LoadingSpinner />
                <span className="text-sm text-muted-foreground">{relatedLoadingMessages[loadingMsgIdx]}</span>
              </div>
            ) : (
              renderSectionList(relatedSections)
            )}
          </div>
        ) : (
          <div className="p-4">
            {isLoadingInsights ? (
              <div className="flex flex-col items-center justify-center mb-4">
                <LoadingSpinner />
                <span className="text-sm text-muted-foreground">{insightsLoadingMessages[loadingMsgIdx]}</span>
              </div>
            ) : (
              <Tabs
                value={activeInsightTab}
                onValueChange={(value) => setActiveInsightTab(value as any)}
              >
                <TabsList className="grid w-full grid-cols-3 gap-2 bg-muted rounded-lg p-1 mb-2">
                  <TabsTrigger value="contradictions" className="data-[state=active]:font-bold text-xs rounded-md px-2 py-1 data-[state=active]:bg-primary/80 data-[state=active]:text-white transition-colors">
                    Contradictions
                  </TabsTrigger>
                  <TabsTrigger value="enhancements" className="data-[state=active]:font-bold text-xs rounded-md px-2 py-1 data-[state=active]:bg-primary/80 data-[state=active]:text-white transition-colors">
                    Enhancements
                  </TabsTrigger>
                  <TabsTrigger value="connections" className="data-[state=active]:font-bold text-xs rounded-md px-2 py-1 data-[state=active]:bg-primary/80 data-[state=active]:text-white transition-colors">
                    Connections
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="contradictions" className="mt-4">
                  {aiInsights.contradictions.length > 0 ? (
                    renderSectionList(aiInsights.contradictions)
                  ) : (
                    <div className="text-muted-foreground italic">
                      No contradictions found.
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="enhancements" className="mt-4">
                  {aiInsights.enhancements.length > 0 ? (
                    renderSectionList(aiInsights.enhancements)
                  ) : (
                    <div className="text-muted-foreground italic">
                      No enhancements found.
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="connections" className="mt-4">
                  {aiInsights.connections.length > 0 ? (
                    renderSectionList(aiInsights.connections)
                  ) : (
                    <div className="text-muted-foreground italic">
                      No connections found.
                    </div>
                  )}
                </TabsContent>
                {/* Podcast Script Display */}
                {aiInsights.podcast_script.length > 0 && (
                  <div className="mt-6">
                    <div className="font-semibold text-base mb-2">
                      Podcast Script
                    </div>
                    <div className="space-y-2">
                      {aiInsights.podcast_script.map((line, idx) => (
                        <div
                          key={idx}
                          className="text-xs text-muted-foreground"
                        >
                          <span className="font-bold">{line.speaker}:</span>{" "}
                          {line.line}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Tabs>
            )}
          </div>
        )}
      </div>

      {/* Audio Format Selection */}
      <div className="p-4 border-t border-panel-border">
        <h3 className="font-medium text-sm mb-3">Choose Podcast Format</h3>
        <div className="grid grid-cols-2 gap-2">
          {audioFormats.map((format) => (
            <Button
              key={format.id}
              variant="outline"
              size="sm"
              onClick={() => handlePodcastClick(format.id)}
              className="flex items-center gap-2 h-auto p-3 hover:bg-primary/10 hover:text-black transition-colors"
              disabled={arePodcastButtonsDisabled}
            >
              <img src={format.img} alt={format.label} className="w-6 h-6 object-contain" />
              <span className="text-xs">{format.label}</span>
              {loadingPersona === format.id && (
                <span className="ml-2 w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>
              )}
            </Button>
          ))}
        </div>
        {/* Loading indicator for podcast data */}
        {isPodcastLoading && (
          <div className="mt-2 text-xs text-muted-foreground text-center">
            Loading podcast data...
          </div>
        )}
      </div>

      {/* Audio Player */}
      {activeAudioFormat && audioUrl && (
        <div className="p-4 border-t border-panel-border bg-card">
          <div className="mb-3">
            <h3 className="font-medium text-sm capitalize">
              {getFormatLabel(activeAudioFormat)}
            </h3>
            <p className="text-xs text-muted-foreground">
              Playing generated podcast audio for the selected text.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              onClick={handlePlayPause}
              size="sm"
              className="w-8 h-8 rounded-full p-0"
              disabled={!audioUrl}
            >
              {isPlaying ? (
                <Pause className="w-3 h-3" />
              ) : (
                <Play className="w-3 h-3" />
              )}
            </Button>

            <div className="flex-1 flex items-center gap-2">
              <span className="text-xs text-muted-foreground min-w-[32px]">
                {formatTime(currentTime)}
              </span>
              <Slider
                value={[currentTime]}
                max={duration || 1}
                step={1}
                onValueChange={handleSeek}
                className="flex-1"
                disabled={!audioUrl}
              />
              <span className="text-xs text-muted-foreground min-w-[32px]">
                {formatTime(duration)}
              </span>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="ml-2 flex items-center justify-center"
              onClick={() => setShowTranscript((prev) => !prev)}
              title={showTranscript ? "Hide Transcript" : "Get Transcript"}
            >
              <FileText className="w-4 h-4" />
            </Button>
          </div>
          {showTranscript && podcastData && podcastData[activeAudioFormat] && (
            <div className="mt-4 p-3 bg-muted rounded text-xs max-h-48 overflow-y-auto custom-scrollbar">
              {podcastData[activeAudioFormat].map((line: string, idx: number) => (
                <div key={idx} className="mb-2">
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RightPanel;
