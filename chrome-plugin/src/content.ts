import { Readability } from '@mozilla/readability';

function cleanDom(doc: Document): void {
  // Remove elements that pollute article text
  const selectorsToRemove = [
    // Comments
    '#comments', '.comments', '.comment-section', '.disqus_thread',
    // Navigation, headers, footers, sidebars
    'nav', 'header', 'footer', 'aside',
    '[role="navigation"]', '[role="complementary"]',
    // Social embeds and sharing
    '.twitter-tweet', '.instagram-media', 'iframe',
    // Ads and promos
    '[class*="advert"]', '[class*="newsletter"]',
    '[class*="signup"]', '[class*="subscribe"]',
    // Related articles
    '[class*="recommended"]', '[class*="more-stories"]',
    '[class*="also-like"]',
    // Media elements
    'video', 'audio', 'svg',
    // Script/style (Readability handles these, but be safe)
    'script', 'style', 'noscript',
  ];

  for (const selector of selectorsToRemove) {
    try {
      doc.querySelectorAll(selector).forEach((el) => el.remove());
    } catch {
      // Invalid selector on this page, skip
    }
  }
}

function cleanExtractedText(text: string): string {
  return text
    // Trim first 100 characters then skip to the next word boundary
    .substring(100).replace(/^\S*\s*/, '')
    // Remove editorial/reporting credits (e.g. "Reporting by X; Editing by Y")
    .replace(/\b(Reporting|Editing|Writing|Compiled|Additional reporting)\s+by\b.*$/gm, '')
    // Remove "Our Standards: ..." trust principle lines
    .replace(/Our Standards:.*$/gm, '')
    // Remove ", opens new tab" link artifacts
    .replace(/,?\s*opens new tab/gi, '')
    // Remove relative timestamps anywhere in text (e.g. "7 hours ago")
    .replace(/\d+\s+(seconds?|minutes?|mins?|hours?|days?|weeks?|months?|years?)\s+ago/gi, '')
    // Remove site names (even inline)
    .replace(/\b(Reuters|BBC|CNN|AP|AFP|Associated Press|Bloomberg)\b/g, '')
    // Remove "Summary" at the start of text or on its own line
    .replace(/^Summary\s*/i, '')
    .replace(/^\s*Summary\s*$/gim, '')
    // Remove photo/image credit lines
    .replace(/^\s*(Getty Images|Reuters|AP Photo|AFP|EPA).*$/gm, '')
    // Ensure space between words that got joined (e.g. "end.Start" -> "end. Start")
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    // Add space after periods/commas/colons not followed by space
    .replace(/([.,:;!?])([A-Za-z])/g, '$1 $2')
    // Collapse multiple spaces into one
    .replace(/ {2,}/g, ' ')
    // Collapse multiple blank lines into one
    .replace(/\n{3,}/g, '\n\n')
    // Remove lines that are just whitespace
    .replace(/^\s+$/gm, '')
    .trim();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'EXTRACT_ARTICLE') {

    try {
      // Clone the document — Readability mutates the DOM it receives
      const documentClone = document.cloneNode(true) as Document;

      // Strip noisy elements before Readability parses
      cleanDom(documentClone);

      const article = new Readability(documentClone).parse();

      if (article) {
        sendResponse({
          success: true,
          title: article.title,
          byline: article.byline,
          content: article.content,
          textContent: cleanExtractedText(article.textContent || ''),
          excerpt: article.excerpt,
          siteName: article.siteName,
          length: article.length,
          lang: article.lang,
        });
      } else {
        sendResponse({
          success: false,
          error: `Readability returned null. URL: ${window.location.href}, title: ${document.title}, body length: ${document.body?.innerHTML.length ?? 0}`,
        });
      }
    } catch (err) {
      sendResponse({
        success: false,
        error: `Exception: ${err instanceof Error ? err.message : String(err)}. URL: ${window.location.href}`,
      });
    }
  }

  // Return true to indicate sendResponse will be called asynchronously
  return true;
});
