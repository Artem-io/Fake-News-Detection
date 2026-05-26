import axios from "axios";

const BASE_URL = "https://Artemi0-fake-news-detector.hf.space";

export interface ModuleDecision {
  module: string;
  real_probability: number;
  effective_weight: number;
  has_data: boolean;
  reasoning: string;
}

export interface LinguisticFlag {
  code: string;
  description: string;
  positive: boolean;
}

export interface LinguisticSignal {
  label: string;
  value: string;
  positive: boolean | null;
}

export interface LinguisticExplanation {
  verdict: string;
  verdict_level: string;
  signals: LinguisticSignal[];
}

export interface Linguistic {
  score: number;
  flags: LinguisticFlag[];
  explanation: LinguisticExplanation;
  headline_consistency: number | null;
}

export interface MatchedSource {
  title: string;
  url: string;
  source: string;
  published_at: string;
  similarity: number;
}

export interface CrossSource {
  verdict: string;
  sources_found: number;
  matched_sources: MatchedSource[];
}

export interface AnalysisResult {
  prediction: string;
  real_probability: number;
  fake_probability: number;
  decisions: ModuleDecision[];
  source_status: string;
  source_domain: string;
  linguistic: Linguistic;
  cross_source: CrossSource;
}

export async function analyzeText(text: string, url: string, title: string = ""): Promise<AnalysisResult> {
  const response = await axios.post<AnalysisResult>(`${BASE_URL}/analyze`, { text, url, title });
  return response.data;
}
