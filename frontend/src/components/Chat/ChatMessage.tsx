import React from 'react';
// @ts-ignore
import userAvatar from '../../assets/user-avatar.svg';
// Using the Moffitt agent logo from public folder
// In React, we need to use process.env.PUBLIC_URL to get the correct public path
const assistantAvatar = `${process.env.PUBLIC_URL}/moffitt-agent-logo.svg`;

interface ChatMessageProps {
  content: string;
  isUser: boolean;
  isWarning?: boolean;
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

  // Enhance code blocks with better formatting
  const withCodeBlocks = withLinks.replace(
    /`([^`]+)`/g,
    '<code>$1</code>'
  );

  return withCodeBlocks
    .split(/\n\n+/)
    .map(p => `<p>${p.replace(/\n/g, '<br/>')}</p>`)
    .join("");
}

const ChatMessage: React.FC<ChatMessageProps> = ({ content, isUser, isWarning }) => {
  const rendered = linkifyHtml(content);
  const messageClass = isUser
    ? 'user-message'
    : isWarning
      ? 'assistant-message warning'
      : 'assistant-message';

  return (
    <div className={`message ${messageClass}`}>
      <div className="message-avatar">
        <img
          src={isUser ? userAvatar : assistantAvatar}
          alt={isUser ? "User" : "Assistant"}
          width="36"
          height="36"
        />
      </div>
      <div className="message-content" dangerouslySetInnerHTML={{ __html: rendered }} />
    </div>
  );
};

export default ChatMessage;

