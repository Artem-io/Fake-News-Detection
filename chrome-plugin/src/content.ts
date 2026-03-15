import { Readability } from '@mozilla/readability';

function cleanDom(doc: Document): void {
  // Remove elements that pollute article text
  const selectorsToRemove = [
    // Comments
    '#comments', '.comments', '.comment-section', '.disqus_thread',
    '[id*="comment"]', '[class*="comment"]',
    // Navigation, headers, footers, sidebars
    'nav', 'header', 'footer', 'aside',
    '[role="navigation"]', '[role="complementary"]', '[role="banner"]',
    // Media captions and figures
    'figcaption', '.wp-caption-text', '.media-caption', '.image-caption',
    '[class*="caption"]',
    // Social embeds and sharing
    '[class*="share"]', '[class*="social"]', '[class*="embed"]',
    '.twitter-tweet', '.instagram-media', 'iframe',
    // Ads and promos
    '[class*="advert"]', '[class*="promo"]', '[class*="newsletter"]',
    '[class*="signup"]', '[class*="subscribe"]', '[id*="ad-"]',
    // Related articles
    '[class*="related"]', '[class*="recommended"]', '[class*="more-stories"]',
    '[class*="read-more"]', '[class*="also-like"]',
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
    // Collapse multiple blank lines into one
    .replace(/\n{3,}/g, '\n\n')
    // Remove lines that are just whitespace
    .replace(/^\s+$/gm, '')
    .trim();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'EXTRACT_ARTICLE') {

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
        error: 'Could not parse article from this page',
      });
    }
  }

  // Return true to indicate sendResponse will be called asynchronously
  return true;
});
