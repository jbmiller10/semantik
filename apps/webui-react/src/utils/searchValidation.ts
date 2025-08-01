/**
 * Validation utilities for search functionality
 * Ensures all input values meet backend requirements and provides user-friendly error messages
 */

export interface ValidationError {
  field: string;
  message: string;
}

export interface SearchValidationRules {
  query: {
    minLength: number;
    maxLength: number;
  };
  topK: {
    min: number;
    max: number;
  };
  scoreThreshold: {
    min: number;
    max: number;
  };
  hybridAlpha: {
    min: number;
    max: number;
  };
  collections: {
    minCount: number;
    maxCount: number;
  };
}

// Default validation rules based on backend schemas
export const DEFAULT_VALIDATION_RULES: SearchValidationRules = {
  query: {
    minLength: 1,
    maxLength: 1000,
  },
  topK: {
    min: 1,
    max: 100,
  },
  scoreThreshold: {
    min: 0.0,
    max: 1.0,
  },
  hybridAlpha: {
    min: 0.0,
    max: 1.0,
  },
  collections: {
    minCount: 1,
    maxCount: 10,
  },
};

/**
 * Validates search query
 */
export function validateQuery(query: string): ValidationError | null {
  const trimmed = query.trim();
  
  if (!trimmed) {
    return { field: 'query', message: 'Search query is required' };
  }
  
  if (trimmed.length < DEFAULT_VALIDATION_RULES.query.minLength) {
    return { 
      field: 'query', 
      message: `Query must be at least ${DEFAULT_VALIDATION_RULES.query.minLength} character` 
    };
  }
  
  if (trimmed.length > DEFAULT_VALIDATION_RULES.query.maxLength) {
    return { 
      field: 'query', 
      message: `Query must not exceed ${DEFAULT_VALIDATION_RULES.query.maxLength} characters` 
    };
  }
  
  return null;
}

/**
 * Validates topK (number of results)
 */
export function validateTopK(topK: number): ValidationError | null {
  if (!Number.isInteger(topK)) {
    return { field: 'topK', message: 'Number of results must be a whole number' };
  }
  
  if (topK < DEFAULT_VALIDATION_RULES.topK.min) {
    return { 
      field: 'topK', 
      message: `Number of results must be at least ${DEFAULT_VALIDATION_RULES.topK.min}` 
    };
  }
  
  if (topK > DEFAULT_VALIDATION_RULES.topK.max) {
    return { 
      field: 'topK', 
      message: `Number of results cannot exceed ${DEFAULT_VALIDATION_RULES.topK.max}` 
    };
  }
  
  return null;
}

/**
 * Validates score threshold
 */
export function validateScoreThreshold(threshold: number): ValidationError | null {
  if (isNaN(threshold)) {
    return { field: 'scoreThreshold', message: 'Score threshold must be a number' };
  }
  
  if (threshold < DEFAULT_VALIDATION_RULES.scoreThreshold.min) {
    return { 
      field: 'scoreThreshold', 
      message: `Score threshold must be at least ${DEFAULT_VALIDATION_RULES.scoreThreshold.min}` 
    };
  }
  
  if (threshold > DEFAULT_VALIDATION_RULES.scoreThreshold.max) {
    return { 
      field: 'scoreThreshold', 
      message: `Score threshold cannot exceed ${DEFAULT_VALIDATION_RULES.scoreThreshold.max}` 
    };
  }
  
  return null;
}

/**
 * Validates hybrid alpha parameter
 */
export function validateHybridAlpha(alpha: number): ValidationError | null {
  if (isNaN(alpha)) {
    return { field: 'hybridAlpha', message: 'Hybrid alpha must be a number' };
  }
  
  if (alpha < DEFAULT_VALIDATION_RULES.hybridAlpha.min) {
    return { 
      field: 'hybridAlpha', 
      message: `Hybrid alpha must be at least ${DEFAULT_VALIDATION_RULES.hybridAlpha.min}` 
    };
  }
  
  if (alpha > DEFAULT_VALIDATION_RULES.hybridAlpha.max) {
    return { 
      field: 'hybridAlpha', 
      message: `Hybrid alpha cannot exceed ${DEFAULT_VALIDATION_RULES.hybridAlpha.max}` 
    };
  }
  
  return null;
}

/**
 * Validates selected collections
 */
export function validateCollections(collections: string[]): ValidationError | null {
  if (!collections || collections.length === 0) {
    return { field: 'collections', message: 'At least one collection must be selected' };
  }
  
  if (collections.length > DEFAULT_VALIDATION_RULES.collections.maxCount) {
    return { 
      field: 'collections', 
      message: `Cannot search more than ${DEFAULT_VALIDATION_RULES.collections.maxCount} collections at once` 
    };
  }
  
  // Validate UUID format for each collection
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const invalidUuids = collections.filter(id => !uuidRegex.test(id));
  
  if (invalidUuids.length > 0) {
    return { 
      field: 'collections', 
      message: 'Invalid collection ID format detected' 
    };
  }
  
  return null;
}

/**
 * Validates all search parameters
 */
export interface SearchValidationParams {
  query: string;
  topK: number;
  scoreThreshold: number;
  hybridAlpha?: number;
  selectedCollections: string[];
  searchType: string;
}

export function validateSearchParams(params: SearchValidationParams): ValidationError[] {
  const errors: ValidationError[] = [];
  
  // Validate query
  const queryError = validateQuery(params.query);
  if (queryError) errors.push(queryError);
  
  // Validate topK
  const topKError = validateTopK(params.topK);
  if (topKError) errors.push(topKError);
  
  // Validate score threshold
  const scoreError = validateScoreThreshold(params.scoreThreshold);
  if (scoreError) errors.push(scoreError);
  
  // Validate hybrid alpha only if using hybrid search
  if (params.searchType === 'hybrid' && params.hybridAlpha !== undefined) {
    const alphaError = validateHybridAlpha(params.hybridAlpha);
    if (alphaError) errors.push(alphaError);
  }
  
  // Validate collections
  const collectionsError = validateCollections(params.selectedCollections);
  if (collectionsError) errors.push(collectionsError);
  
  return errors;
}

/**
 * Sanitizes and normalizes search query
 */
export function sanitizeQuery(query: string): string {
  return query.trim().replace(/\s+/g, ' ');
}

/**
 * Clamps a numeric value within bounds
 */
export function clampValue(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Formats validation errors for display
 */
export function formatValidationErrors(errors: ValidationError[]): string {
  if (errors.length === 0) return '';
  if (errors.length === 1) return errors[0].message;
  
  return errors.map(e => `• ${e.message}`).join('\n');
}