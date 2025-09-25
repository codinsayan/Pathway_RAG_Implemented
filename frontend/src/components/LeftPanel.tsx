import { useState } from "react";
import { Search, FileText, Mic, Trash2, Check } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import LoadingText from "./LoadingText";

interface Document {
  id: string;
  name: string;
  type: "business" | "market";
  sections?: Section[];
}

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
  // legacy fields for savedSections
  id?: string;
  title?: string;
  preview?: string;
  source?: string;
  hasGlow?: boolean;
}

interface LeftPanelProps {
  onSectionClick: (section: Section) => void;
  onDocumentSelect: (documentId: string) => void;
  selectedDocument?: string;
}

const LeftPanel = ({ onSectionClick, onDocumentSelect, selectedDocument }: LeftPanelProps) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Fetch PDF list from backend
  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://localhost:8000/list_pdfs");
      const data = await response.json();
      if (data.pdfs) {
        // For each document, fetch its related sections
        const docsWithSections = await Promise.all(
          data.pdfs.map(async (filename: string) => {
            let sections: any[] = [];
            try {
              const secRes = await fetch("http://localhost:8000/get_retrieved_sections", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ document_name: filename })
              });
              const secData = await secRes.json();
              sections = secData.retrieved_sections || [];
            } catch (err) {
              sections = [];
            }
            return {
              id: filename,
              name: filename,
              type: "business",
              sections: sections
            };
          })
        );
        setDocuments(docsWithSections);
      }
    } catch (error) {
      console.error("Error fetching PDFs:", error);
      setDocuments([]);
    }
  };

  // Initial fetch
  useState(() => {
    fetchDocuments();
  });

  // Handle batch upload
  const handleUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFiles || selectedFiles.length === 0) return;
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }
    try {
      const response = await fetch("http://localhost:8000/upload_batch", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.filenames) {
        // setUploadSuccess("Upload successful!");
        fetchDocuments();
      } else {
        setUploadError("Upload failed. Please try again.");
      }
    } catch (error) {
      console.error("Error uploading PDFs:", error);
      setUploadError("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  // Minimal document upload section, simple full-width file picker
  const renderDocumentUpload = () => (
    <div className="p-3 border-b bg-card flex flex-col gap-2 min-w-0 rounded-b-md">
      <form className="flex flex-col sm:flex-row gap-2 items-center min-w-0 w-full" onSubmit={e => e.preventDefault()}>
        <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-primary w-full">
          <input
            type="file"
            accept="application/pdf"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <span className="px-4 py-2 w-full text-center bg-muted text-foreground rounded-lg border border-border font-semibold text-xs tracking-wide">
            Choose PDFs
          </span>
        </label>
      </form>
      {selectedFiles && selectedFiles.length > 0 && (
        <div className="text-xs text-muted-foreground mt-1">{selectedFiles.length} file(s) selected</div>
      )}
      {uploadError && <div className="text-red-500 text-xs mt-2">{uploadError}</div>}
      {uploadSuccess && <div className="text-green-600 text-xs mt-2">{uploadSuccess}</div>}
    </div>
  );

  // Upload progress steps
  const uploadSteps = [
    "Yeeting files into the cloud!",
    "Chopping docs like a word ninja!",
    "Turning text into mathy gibberish!",
    "Building a disco den for data!",
    "Cramming vectors into a cosmic vault!",
    "Connecting the dots..."
  ];
  const [uploadStep, setUploadStep] = useState(0);

  // Modified handleFileChange to show stepwise progress with 1s delay, independent of actual upload
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(event.target.files);
    setUploadError(null);
    setUploadSuccess(null);
    if (event.target.files && event.target.files.length > 0) {
      setUploading(true);
      setUploadStep(0);
      // Animate step text
      let step = 0;
      const stepInterval = setInterval(() => {
        step++;
        if (step < uploadSteps.length) {
          setUploadStep(step);
        } else {
          clearInterval(stepInterval);
        }
      }, 3500);
      // Actual upload
      (async () => {
        const formData = new FormData();
        for (let i = 0; i < event.target.files.length; i++) {
          formData.append("files", event.target.files[i]);
        }
        try {
          const response = await fetch("http://localhost:8000/upload_batch", {
            method: "POST",
            body: formData
          });
          const data = await response.json();
          if (data.filenames) {
            fetchDocuments();
            setUploadSuccess("Upload successful!");
          } else {
            setUploadError("Upload failed. Please try again.");
          }
        } catch (error) {
          setUploadError("Upload failed. Please try again.");
        } finally {
          setUploading(false);
          setUploadStep(0);
          clearInterval(stepInterval);
        }
      })();
    }
  };

  const getDocumentIcon = (type: string, isSelected: boolean) => {
    if (isSelected) {
      return <FileText className="w-4 h-4 text-blue-600" />;
    }
    switch (type) {
      case "business":
        return <FileText className="w-4 h-4 text-gray-500" />;
      case "market":
        return <FileText className="w-4 h-4 text-blue-500" />;
      default:
        return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="h-full flex flex-col panel border-r">
      {/* Upload PDFs */}
      {renderDocumentUpload()}

      {/* Tab Headers */}
      <div className="flex border-b border-panel-border">
        <div className="flex-1 px-4 py-3 text-lg font-bold text-left">Library</div>
      </div>


      {/* Content */}
      <div className="flex-1 overflow-auto custom-scrollbar">
        <div className="p-4 space-y-4">
          {uploading && (
            <div className="mb-2 flex items-center gap-2 text-primary text-sm">
              <span className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>
              {uploadSteps[uploadStep]}
            </div>
          )}
          {documents.map((doc) => (
            <div key={doc.id} className="space-y-2 flex items-center">
              <button
                onClick={() => onDocumentSelect(doc.id)}
                className={`flex items-center gap-3 flex-1 p-3 rounded-lg hover:bg-sidebar-hover transition-colors text-left ${doc.id === selectedDocument ? 'bg-primary/10 border border-primary' : ''}`}
                style={doc.id === selectedDocument ? { fontWeight: 'bold' } : {}}
              >
                {getDocumentIcon(doc.type, doc.id === selectedDocument)}
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate max-w-[180px]" title={doc.name}>
                    {doc.name.length > 32 ? doc.name.slice(0, 29) + '...' : doc.name}
                  </div>
                </div>
                <button
                  className={`ml-2 px-2 py-1 text-xs text-red-600 border border-red-200 rounded hover:bg-red-50 flex items-center gap-1 ${confirmDeleteId === doc.id ? 'bg-green-100 font-bold' : ''}`}
                  title="Delete document"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (confirmDeleteId !== doc.id) {
                      setConfirmDeleteId(doc.id);
                      return;
                    }
                    try {
                      const response = await fetch("http://localhost:8000/delete_document", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ document_name: doc.name })
                      });
                      const data = await response.json();
                      if (!data.error) {
                        fetchDocuments();
                      }
                    } catch (err) {
                    }
                    setConfirmDeleteId(null);
                  }}
                >
                  {confirmDeleteId === doc.id ? (
                    <Check className="w-4 h-4 text-green-600" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </button>

              {/* Related Sections for this document */}
              {doc.sections && doc.sections.length > 0 && (
                <div className="ml-6 space-y-2">
                  {doc.sections.map((section, idx) => (
                    <button
                      key={idx}
                      className="w-full p-3 text-left bg-card border border-border rounded-lg hover:shadow-sm transition-all duration-200"
                      onClick={() => onSectionClick({
                        ...section,
                        document_name: doc.id,
                      })}
                    >
                      <div className="flex flex-col min-w-0">
                        <div className="font-medium text-xs mb-1">
                          {section.section_title || section.title}
                        </div>
                        {section.original_content && section.original_content.trim() !== '' && (
                          <div className="text-xs text-foreground mb-2 line-clamp-3">
                            {section.original_content.length > 120
                              ? section.original_content.slice(0, 120) + '...'
                              : section.original_content}
                          </div>
                        )}
                        <div className="text-xs text-muted-foreground mb-2 line-clamp-2">
                          {section.full_path || section.preview}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {section.page_number !== undefined ? `Page: ${section.page_number}` : ''}
                        </div>
                        <div className="text-right mt-2">
                          <span className="text-2xl">"</span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LeftPanel;
