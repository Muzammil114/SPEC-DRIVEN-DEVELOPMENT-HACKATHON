import React, { useState, useEffect, useRef } from 'react';
import { Box, TextField, Button, Paper, Typography, List, ListItem, ListItemText, CircularProgress } from '@mui/material';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sources?: Array<{
    id: string;
    text: string;
    metadata: any;
  }>;
}

const ChatBot: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your Physical AI & Humanoid Robotics tutor. Ask me any questions about the textbook content.',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Call backend API
      const response = await fetch('/api/v1/chat/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputText,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add bot response
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources,
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 800, height: '600px', display: 'flex', flexDirection: 'column', border: '1px solid #ccc', borderRadius: 2 }}>
      <Paper elevation={3} sx={{ p: 2, flexGrow: 1, overflowY: 'auto' }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Physical AI & Humanoid Robotics Tutor
        </Typography>

        <List>
          {messages.map((message) => (
            <ListItem
              key={message.id}
              alignItems="flex-start"
              sx={{
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                pb: 0,
              }}
            >
              <Paper
                sx={{
                  p: 1.5,
                  maxWidth: '80%',
                  backgroundColor: message.sender === 'user' ? '#e3f2fd' : '#f5f5f5',
                  borderRadius: message.sender === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                }}
              >
                <ListItemText
                  primary={message.text}
                  primaryTypographyProps={{
                    variant: 'body2',
                    sx: { wordWrap: 'break-word', whiteSpace: 'normal' }
                  }}
                />

                {message.sources && message.sources.length > 0 && (
                  <Box sx={{ mt: 1, pt: 1, borderTop: '1px solid #ddd' }}>
                    <Typography variant="caption" color="textSecondary">
                      Sources:
                    </Typography>
                    {message.sources.slice(0, 2).map((source, idx) => (
                      <Typography key={idx} variant="caption" display="block" color="textSecondary">
                        - {source.metadata?.title || 'Reference'}
                      </Typography>
                    ))}
                  </Box>
                )}
              </Paper>
            </ListItem>
          ))}
          {isLoading && (
            <ListItem alignItems="flex-start" sx={{ justifyContent: 'flex-start', pb: 0 }}>
              <Paper
                sx={{
                  p: 1.5,
                  maxWidth: '80%',
                  backgroundColor: '#f5f5f5',
                  borderRadius: '18px 18px 18px 4px',
                }}
              >
                <CircularProgress size={20} />
              </Paper>
            </ListItem>
          )}
          <div ref={messagesEndRef} />
        </List>
      </Paper>

      <Box sx={{ p: 2, display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about Physical AI & Humanoid Robotics..."
          variant="outlined"
          size="small"
        />
        <Button
          variant="contained"
          onClick={handleSendMessage}
          disabled={isLoading || !inputText.trim()}
          sx={{ height: 'fit-content' }}
        >
          Send
        </Button>
      </Box>
    </Box>
  );
};

export default ChatBot;