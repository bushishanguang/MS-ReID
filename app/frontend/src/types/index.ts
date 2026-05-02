export type VisualResult = {
  image_name: string;
  original?: string | null;
  LH?: string | null;
  HL?: string | null;
  HH?: string | null;
  attention_overlay?: string | null;
  attention_weights?: {
    branch_1x1: number;
    branch_3x3: number;
    branch_5x5: number;
  } | null;
  message?: string | null;
};

export type ExperimentResult = {
  experiment: string;
  rank1: number;
  map: number;
  rank1_delta: number;
  map_delta: number;
  is_best_rank1: boolean;
  is_best_map: boolean;
};

export type ExperimentCurve = {
  experiment: string;
  epochs: number[];
  accuracy: number[];
  loss: number[];
};

export type RetrievalResult = {
  rank: number;
  image_name: string;
  image_url: string;
  score: number;
};

export type RetrievalResponse = {
  query: string;
  top_k: number;
  results: RetrievalResult[];
};
