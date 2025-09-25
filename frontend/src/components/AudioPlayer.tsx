import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, X, SkipBack, SkipForward } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

interface AudioPlayerProps {
  isOpen: boolean;
  onClose: () => void;
  audioFormat: 'debater' | 'investigator' | 'fundamentals' | 'connections' | null;
  content: string;
}

const AudioPlayer = ({ isOpen, onClose, audioFormat, content }: AudioPlayerProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(80);
  const [currentText, setCurrentText] = useState('');
  const audioRef = useRef<HTMLAudioElement>(null);

  // Mock text that would sync with audio
  const textSegments = [
    "Lorem Ipsum is simply dummy text",
    "of the printing and typesetting industry.",
    "Lorem Ipsum has been the industry's",
    "standard dummy text ever since the 1500s..."
  ];

  useEffect(() => {
    if (audioFormat && content) {
      // Here you would integrate with Azure TTS
      // For now, we'll simulate audio generation
      console.log(`Generating audio for ${audioFormat}: ${content}`);
      setDuration(60); // Mock 60 second duration
    }
  }, [audioFormat, content]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          const newTime = prev + 1;
          // Simulate text highlighting based on time
          const segmentIndex = Math.floor((newTime / duration) * textSegments.length);
          setCurrentText(textSegments[segmentIndex] || '');
          
          if (newTime >= duration) {
            setIsPlaying(false);
            return duration;
          }
          return newTime;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, duration]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (value: number[]) => {
    const newTime = value[0];
    setCurrentTime(newTime);
    const segmentIndex = Math.floor((newTime / duration) * textSegments.length);
    setCurrentText(textSegments[segmentIndex] || '');
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getFormatIcon = () => {
    switch (audioFormat) {
      case 'debater':
        return '💬';
      case 'investigator':
        return '🔍';
      case 'fundamentals':
        return '🧠';
      case 'connections':
        return '🔗';
      default:
        return '🎵';
    }
  };

  return (
    <div className={`audio-player ${isOpen ? 'open' : ''}`}>
      <div className="p-4 bg-card border-t border-border">
        {/* Audio Text Display */}
        <div className="mb-4 p-3 bg-muted rounded-lg min-h-[60px] flex items-center">
          <p className="text-sm text-center w-full">
            {currentText || 'Audio text will appear here as it plays...'}
          </p>
        </div>

        {/* Player Controls */}
        <div className="flex items-center gap-4">
          {/* Format Icon */}
          <div className="flex items-center gap-2">
            <span className="text-xl">{getFormatIcon()}</span>
            <span className="text-sm font-medium capitalize">
              {audioFormat || 'Audio'}
            </span>
          </div>

          {/* Transport Controls */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm">
              <SkipBack className="w-4 h-4" />
            </Button>
            
            <Button 
              onClick={handlePlayPause}
              size="sm"
              className="w-10 h-10 rounded-full"
            >
              {isPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </Button>
            
            <Button variant="ghost" size="sm">
              <SkipForward className="w-4 h-4" />
            </Button>
          </div>

          {/* Progress */}
          <div className="flex-1 flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {formatTime(currentTime)}
            </span>
            <Slider
              value={[currentTime]}
              max={duration}
              step={1}
              onValueChange={handleSeek}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground">
              {formatTime(duration)}
            </span>
          </div>

          {/* Volume */}
          <div className="flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-muted-foreground" />
            <Slider
              value={[volume]}
              max={100}
              step={1}
              onValueChange={(value) => setVolume(value[0])}
              className="w-20"
            />
          </div>

          {/* Close */}
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AudioPlayer;