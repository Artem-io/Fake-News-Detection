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
    // Remove editorial/reporting credits
    .replace(/\b(Reporting|Editing|Writing|Compiled|Additional reporting)\s+by\b.*$/gm, '')
    // Remove trust principle lines
    .replace(/Our Standards:.*$/gm, '')
    // Remove ", opens new tab" link artifacts
    .replace(/,?\s*opens new tab/gi, '')
    // Remove relative timestamps (e.g. "7 hours ago")
    .replace(/\d+\s+(seconds?|minutes?|mins?|hours?|days?|weeks?|months?|years?)\s+ago/gi, '')
    // Remove site names
    .replace(/\b(Reuters|BBC|CNN|AP|AFP|Associated Press|Bloomberg)\b/g, '')
    // Remove "Summary" at the start of text or on its own line
    .replace(/^Summary\s*/i, '')
    .replace(/^\s*Summary\s*$/gim, '')
    // Remove photo/image credit lines
    .replace(/^\s*(Getty Images|Reuters|AP Photo|AFP|EPA).*$/gm, '')
    // Ensure space between words that got joined
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    // Add space after periods/commas not followed by space
    .replace(/([.,:;!?])([A-Za-z])/g, '$1 $2')
    // Collapse multiple spaces into one
    .replace(/ {2,}/g, ' ')
    // Collapse multiple blank lines into one
    .replace(/\n{3,}/g, '\n\n')
    // Remove lines that are just whitespace
    .replace(/^\s+$/gm, '')
    .trim();
}

function paragraphFallback(): string {
  // Fallback: collect all <p> text longer than 40 chars
  return Array.from(document.querySelectorAll('p'))
    .map((p) => p.textContent?.trim() ?? '')
    .filter((t) => t.length > 40)
    .join('\n\n');
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'EXTRACT_ARTICLE') {

    try {
      // Attempt 1: Readability with pre-cleaned DOM
      const cleanedClone = document.cloneNode(true) as Document;
      cleanDom(cleanedClone);
      let article = new Readability(cleanedClone).parse();

      // Attempt 2: Readability on raw DOM — cleanDom may have stripped content the
      if (!article) {
        const rawClone = document.cloneNode(true) as Document;
        article = new Readability(rawClone).parse();
      }

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
        return true;
      }

      // Attempt 3: manual paragraph extraction
      const fallbackText = paragraphFallback();
      if (fallbackText.length > 200) {
        sendResponse({
          success: true,
          title: document.title,
          byline: '',
          content: '',
          textContent: cleanExtractedText(fallbackText),
          excerpt: '',
          siteName: '',
          length: fallbackText.length,
          lang: document.documentElement.lang || '',
        });
        return true;
      }

      sendResponse({
        success: false,
        error: 'Could not extract article content from this page.',
      });
    } catch (err) {
      sendResponse({
        success: false,
        error: `Exception: ${err instanceof Error ? err.message : String(err)}. URL: ${window.location.href}`,
      });
    }
  }

  return true;
});
