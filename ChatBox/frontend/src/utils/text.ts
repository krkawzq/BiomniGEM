export function addLineNumbers(text: string): string {
  const lines = text.split('\n');
  const maxLineNumber = lines.length;
  const padding = maxLineNumber.toString().length;
  
  return lines
    .map((line, index) => {
      const lineNumber = (index + 1).toString().padStart(padding, ' ');
      return `${lineNumber}| ${line}`;
    })
    .join('\n');
}

export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}
