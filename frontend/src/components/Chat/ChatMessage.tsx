import React from 'react';

interface ChatMessageProps {
  content: string;
  isUser: boolean;
}

// Convert plain text to basic HTML: linkify URLs and preserve line breaks.
function linkifyHtml(text: string): string {
  // If it already contains HTML tags, trust backend and return as-is
  if (/<\w+\b[^>]*>/i.test(text)) return text;

  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const withLinks = escaped.replace(
    /(https?:\/\/[^\s)]+)(?=[\s)|]|$)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>'
  );

  return withLinks
    .split(/\n\n+/)
    .map(p => `<p>${p.replace(/\n/g, '<br/>')}</p>`)
    .join("");
}

const ChatMessage: React.FC<ChatMessageProps> = ({ content, isUser }) => {
  const rendered = linkifyHtml(content);
  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-avatar">
        {isUser ? '🙂' : '🤖'}
      </div>
      <div className="message-content" dangerouslySetInnerHTML={{ __html: rendered }} />
    </div>
  );
};

export default ChatMessage;

