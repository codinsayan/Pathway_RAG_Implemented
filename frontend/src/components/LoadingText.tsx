import { useEffect, useState } from 'react';

const LoadingText = () => {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => {
        if (prev === '...') return '';
        return prev + '.';
      });
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center justify-center p-8 text-muted-foreground">
      <span className="animate-pulse">connecting the dots{dots}</span>
    </div>
  );
};

export default LoadingText;