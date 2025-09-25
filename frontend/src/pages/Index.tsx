import { useState } from "react";
import axios from "axios";
import ResizablePanel from "@/components/ResizablePanel";
import LeftPanel from "@/components/LeftPanel";
import PDFViewer from "@/components/PDFViewer";
import RightPanel from "@/components/RightPanel";

interface Section {
  id?: string;
  title?: string;
  preview?: string;
  source?: string;
  section_title?: string;
  original_content?: string;
  document_name?: string;
  full_path?: string;
}

const Index = () => {
  const [selectedDocument, setSelectedDocument] = useState<string>("");
  const [selectedText, setSelectedText] = useState<string>("");
  const [audioFormat, setAudioFormat] = useState<
    "debater" | "investigator" | "fundamentals" | "connections" | null
  >(null);
  const [searchOnLoad, setSearchOnLoad] = useState<string>("");

  // When a section is clicked, open its document and search for its heading/content
  const handleSectionClick = (section: Section) => {
    // Prefer section.section_title or section.title for heading, fallback to original_content
    const heading = section.section_title || section.title || section.original_content || "";
    // Prefer document_name or full_path for document id
    const docId = section.document_name || section.full_path || "";
    setSelectedDocument(docId);
    setSearchOnLoad(heading);
    console.log(heading)
    setSelectedText(heading); // Optionally set selected text for right panel
  };

  const handleDocumentSelect = (documentId: string) => {
    setSelectedDocument(documentId);
    setSearchOnLoad(""); // Clear auto-search when just switching document
    console.log("Document selected:", documentId);
  };

  const handleTextSelect = (text: string) => {
    setSelectedText(text);
  };

  const handleAudioFormatSelect = (
    format: "debater" | "investigator" | "fundamentals" | "connections"
  ) => {
    setAudioFormat(format);
    console.log("Audio format selected:", format);
  };

  return (
    <div className="h-screen flex bg-background overflow-hidden">
      {/* Left Panel */}
      <ResizablePanel
        defaultWidth={340}
        minWidth={340}
        maxWidth={500}
        position="left"
        collapsible={true}
      >
        <LeftPanel
          onSectionClick={handleSectionClick}
          onDocumentSelect={handleDocumentSelect}
          selectedDocument={selectedDocument}
        />
      </ResizablePanel>

      {/* Middle Panel - PDF Viewer */}
      <div className="flex-1 h-full">
        <PDFViewer
          documentId={selectedDocument}
          onTextSelect={handleTextSelect}
          searchOnLoad={searchOnLoad}
        />
      </div>

      {/* Right Panel */}
      <ResizablePanel
        defaultWidth={360}
        minWidth={360}
        maxWidth={500}
        position="right"
        collapsible={true}
      >
        <RightPanel
          selectedText={selectedText}
          onAudioFormatSelect={handleAudioFormatSelect}
          activeAudioFormat={audioFormat}
          onSectionCardClick={(docId: string, searchTerm: string) => {
            if (docId === selectedDocument) {
              setSearchOnLoad(searchTerm);
            } else {
              setSelectedDocument(docId);
              setSearchOnLoad(searchTerm);
            }
          }}
        />
      </ResizablePanel>
    </div>
  );
};

export default Index;
