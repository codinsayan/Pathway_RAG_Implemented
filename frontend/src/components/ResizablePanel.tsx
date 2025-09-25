import { useState, useRef, useEffect, ReactNode } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ResizablePanelProps {
  children: ReactNode;
  defaultWidth: number;
  minWidth: number;
  maxWidth: number;
  position: 'left' | 'right';
  onResize?: (width: number) => void;
  collapsible?: boolean;
}

const ResizablePanel = ({ 
  children, 
  defaultWidth, 
  minWidth, 
  maxWidth, 
  position,
  onResize,
  collapsible = false
}: ResizablePanelProps) => {
  const [width, setWidth] = useState(defaultWidth);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);

  const currentWidth = isCollapsed ? 0 : width;

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const containerRect = panelRef.current?.parentElement?.getBoundingClientRect();
      if (!containerRect) return;

      let newWidth;
      if (position === 'left') {
        newWidth = e.clientX - containerRect.left;
      } else {
        newWidth = containerRect.right - e.clientX;
      }

      newWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      setWidth(newWidth);
      onResize?.(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, minWidth, maxWidth, position, onResize]);

  const handleMouseDown = () => {
    setIsResizing(true);
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <div
      ref={panelRef}
      className={`relative h-full ${isResizing ? '' : 'transition-all duration-300'}`}
      style={{ width: `${currentWidth}px`, flexShrink: 0 }}
    >
      {/* Collapse Button - positioned on resizer */}
      {collapsible && !isCollapsed && (
        <Button
          variant="ghost"
          size="sm"
          className={`absolute top-1/2 -translate-y-1/2 z-30 w-6 h-8 p-0 bg-background/80 backdrop-blur-sm border border-border/50 hover:bg-accent ${
            position === 'left' ? '-right-3' : '-left-3'
          }`}
          onClick={toggleCollapse}
        >
          {position === 'left' ? (
            <ChevronLeft className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </Button>
      )}
      
      {/* Collapsed state expand button */}
      {collapsible && isCollapsed && (
        <Button
          variant="ghost"
          size="sm"
          className={`absolute top-1/2 -translate-y-1/2 z-30 w-6 h-8 p-0 bg-background/80 backdrop-blur-sm border border-border/50 hover:bg-accent ${
            position === 'left' ? '-right-6' : '-left-6'
          }`}
          onClick={toggleCollapse}
        >
          {position === 'left' ? (
            <ChevronRight className="h-3 w-3" />
          ) : (
            <ChevronLeft className="h-3 w-3" />
          )}
        </Button>
      )}

      {/* Panel Content */}
      <div className={`h-full transition-opacity duration-300 ${isCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        {children}
      </div>
      
      {/* Resizer Handle */}
      {!isCollapsed && (
        <div
          ref={resizerRef}
          className={`absolute top-0 w-1 h-full resizer z-10 ${
            position === 'left' ? 'right-0' : 'left-0'
          }`}
          onMouseDown={handleMouseDown}
        />
      )}
    </div>
  );
};

export default ResizablePanel;