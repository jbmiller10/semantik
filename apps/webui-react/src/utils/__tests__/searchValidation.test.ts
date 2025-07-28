import { 
  validateQuery, 
  validateTopK, 
  validateScoreThreshold, 
  validateHybridAlpha,
  validateCollections,
  validateSearchParams,
  sanitizeQuery,
  clampValue
} from '../searchValidation';

describe('Search Validation Utilities', () => {
  describe('validateQuery', () => {
    it('should accept valid queries', () => {
      expect(validateQuery('test query')).toBeNull();
      expect(validateQuery('a')).toBeNull();
      expect(validateQuery('  test  ')).toBeNull(); // trimmed
    });

    it('should reject empty queries', () => {
      expect(validateQuery('')).toEqual({
        field: 'query',
        message: 'Search query is required'
      });
      expect(validateQuery('   ')).toEqual({
        field: 'query',
        message: 'Search query is required'
      });
    });

    it('should reject queries exceeding max length', () => {
      const longQuery = 'a'.repeat(1001);
      expect(validateQuery(longQuery)).toEqual({
        field: 'query',
        message: 'Query must not exceed 1000 characters'
      });
    });
  });

  describe('validateTopK', () => {
    it('should accept valid topK values', () => {
      expect(validateTopK(1)).toBeNull();
      expect(validateTopK(50)).toBeNull();
      expect(validateTopK(100)).toBeNull();
    });

    it('should reject non-integer values', () => {
      expect(validateTopK(1.5)).toEqual({
        field: 'topK',
        message: 'Number of results must be a whole number'
      });
    });

    it('should reject values outside bounds', () => {
      expect(validateTopK(0)).toEqual({
        field: 'topK',
        message: 'Number of results must be at least 1'
      });
      expect(validateTopK(101)).toEqual({
        field: 'topK',
        message: 'Number of results cannot exceed 100'
      });
    });
  });

  describe('validateScoreThreshold', () => {
    it('should accept valid score thresholds', () => {
      expect(validateScoreThreshold(0)).toBeNull();
      expect(validateScoreThreshold(0.5)).toBeNull();
      expect(validateScoreThreshold(1)).toBeNull();
    });

    it('should reject values outside bounds', () => {
      expect(validateScoreThreshold(-0.1)).toEqual({
        field: 'scoreThreshold',
        message: 'Score threshold must be at least 0'
      });
      expect(validateScoreThreshold(1.1)).toEqual({
        field: 'scoreThreshold',
        message: 'Score threshold cannot exceed 1'
      });
    });
  });

  describe('validateHybridAlpha', () => {
    it('should accept valid hybrid alpha values', () => {
      expect(validateHybridAlpha(0)).toBeNull();
      expect(validateHybridAlpha(0.7)).toBeNull();
      expect(validateHybridAlpha(1)).toBeNull();
    });

    it('should reject values outside bounds', () => {
      expect(validateHybridAlpha(-0.1)).toEqual({
        field: 'hybridAlpha',
        message: 'Hybrid alpha must be at least 0'
      });
      expect(validateHybridAlpha(1.1)).toEqual({
        field: 'hybridAlpha',
        message: 'Hybrid alpha cannot exceed 1'
      });
    });
  });

  describe('validateCollections', () => {
    it('should accept valid collection UUIDs', () => {
      const validUuids = [
        '123e4567-e89b-12d3-a456-426614174000',
        '456e7890-e89b-12d3-a456-426614174001'
      ];
      expect(validateCollections(validUuids)).toBeNull();
    });

    it('should reject empty collections', () => {
      expect(validateCollections([])).toEqual({
        field: 'collections',
        message: 'At least one collection must be selected'
      });
    });

    it('should reject too many collections', () => {
      const tooMany = Array(11).fill('123e4567-e89b-12d3-a456-426614174000');
      expect(validateCollections(tooMany)).toEqual({
        field: 'collections',
        message: 'Cannot search more than 10 collections at once'
      });
    });

    it('should reject invalid UUIDs', () => {
      const invalidUuids = ['not-a-uuid', '123e4567-e89b-12d3-a456-426614174000'];
      expect(validateCollections(invalidUuids)).toEqual({
        field: 'collections',
        message: 'Invalid collection ID format detected'
      });
    });
  });

  describe('validateSearchParams', () => {
    it('should return no errors for valid params', () => {
      const validParams = {
        query: 'test query',
        topK: 10,
        scoreThreshold: 0.5,
        hybridAlpha: 0.7,
        selectedCollections: ['123e4567-e89b-12d3-a456-426614174000'],
        searchType: 'hybrid'
      };
      expect(validateSearchParams(validParams)).toEqual([]);
    });

    it('should collect all validation errors', () => {
      const invalidParams = {
        query: '',
        topK: 200,
        scoreThreshold: -1,
        hybridAlpha: 2,
        selectedCollections: [],
        searchType: 'hybrid'
      };
      const errors = validateSearchParams(invalidParams);
      expect(errors).toHaveLength(5);
      expect(errors.map(e => e.field)).toEqual([
        'query',
        'topK',
        'scoreThreshold',
        'hybridAlpha',
        'collections'
      ]);
    });
  });

  describe('sanitizeQuery', () => {
    it('should trim and normalize whitespace', () => {
      expect(sanitizeQuery('  test  query  ')).toBe('test query');
      expect(sanitizeQuery('test\n\nquery')).toBe('test query');
      expect(sanitizeQuery('test\t\tquery')).toBe('test query');
    });
  });

  describe('clampValue', () => {
    it('should clamp values within bounds', () => {
      expect(clampValue(5, 0, 10)).toBe(5);
      expect(clampValue(-5, 0, 10)).toBe(0);
      expect(clampValue(15, 0, 10)).toBe(10);
    });
  });
});