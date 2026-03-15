import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

export interface SourceVerification {
  domain: string;
  status: string;
  label: number;
}

export interface FactCheckResult {
  claim_text: string;
  found: boolean;
  rating: string;
  publisher: string;
  url: string;
  title: string;
  textual_rating: string;
}

export interface FactCheck {
  claims_extracted: number;
  claims_checked: number;
  claims_with_results: number;
  overall_score: number;
  results: FactCheckResult[];
}

export interface LinguisticAnalysis {
  score: number;
  confidence: number;
  flags: string[];
  explanation: string;
  details: Record<string, unknown>;
}

export interface AnalysisResult {
  prediction: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  source_verification: SourceVerification;
  fact_check: FactCheck;
  linguistic_analysis: LinguisticAnalysis;
}

export async function analyzeText(text: string, url: string): Promise<AnalysisResult> {
  const response = await axios.post<AnalysisResult>(`${BASE_URL}/analyze`, { text, url });
  return response.data;
}