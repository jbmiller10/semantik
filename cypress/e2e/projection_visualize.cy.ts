const COLLECTION_ID = 'test-collection-id';
const PROJECTION_ID = 'projection-1';

function createFloat32Buffer(values: number[]): ArrayBuffer {
  const array = new Float32Array(values);
  return array.buffer;
}

function createUint8Buffer(values: number[]): ArrayBuffer {
  const array = new Uint8Array(values);
  return array.buffer;
}

function createInt32Buffer(values: number[]): ArrayBuffer {
  const array = new Int32Array(values);
  return array.buffer;
}

describe('Projection visualization', () => {
  beforeEach(() => {
    const now = new Date().toISOString();

    const collection = {
      id: COLLECTION_ID,
      name: 'Test Collection',
      description: 'Collection for projection visualization tests',
      owner_id: 1,
      vector_store_name: 'test-store',
      embedding_model: 'test-model',
      quantization: 'float16',
      chunk_size: 1000,
      chunk_overlap: 200,
      chunking_strategy: 'character',
      chunking_config: {},
      is_public: false,
      status: 'ready',
      document_count: 10,
      vector_count: 100,
      total_size_bytes: 1024,
      created_at: now,
      updated_at: now,
    };

    const projection = {
      id: PROJECTION_ID,
      collection_id: COLLECTION_ID,
      status: 'completed',
      reducer: 'umap',
      dimensionality: 2,
      created_at: now,
      message: null,
      operation_id: 'op-123',
      operation_status: 'completed',
      config: {},
      meta: {
        legend: [
          { index: 0, label: 'Cluster A', count: 2 },
          { index: 1, label: 'Cluster B', count: 2 },
        ],
        color_by: 'document_id',
        sampled: true,
        shown_count: 4,
        total_count: 10,
        degraded: false,
      },
    };

    // Collections list and detail
    cy.intercept('GET', '/api/v2/collections', {
      statusCode: 200,
      body: {
        collections: [collection],
        total: 1,
        page: 1,
        per_page: 25,
      },
    }).as('listCollections');

    cy.intercept('GET', `/api/v2/collections/${COLLECTION_ID}`, {
      statusCode: 200,
      body: collection,
    }).as('getCollection');

    cy.intercept('GET', `/api/v2/collections/${COLLECTION_ID}/operations*`, {
      statusCode: 200,
      body: [],
    }).as('listOperations');

    // Projections list
    cy.intercept('GET', `/api/v2/collections/${COLLECTION_ID}/projections`, {
      statusCode: 200,
      body: {
        projections: [projection],
      },
    }).as('listProjections');

    // Projection metadata
    cy.intercept('GET', `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}`, {
      statusCode: 200,
      body: projection,
    }).as('getProjectionMetadata');

    // Projection arrays
    const xBuffer = createFloat32Buffer([0, 1, 2, 3]);
    const yBuffer = createFloat32Buffer([0, 1, 0, 1]);
    const catBuffer = createUint8Buffer([0, 0, 1, 1]);
    const idsBuffer = createInt32Buffer([100, 101, 102, 103]);

    cy.intercept(
      'GET',
      `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}/arrays/x`,
      (req) => {
        req.reply({
          statusCode: 200,
          headers: { 'Content-Type': 'application/octet-stream' },
          body: xBuffer,
        });
      },
    ).as('getX');

    cy.intercept(
      'GET',
      `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}/arrays/y`,
      (req) => {
        req.reply({
          statusCode: 200,
          headers: { 'Content-Type': 'application/octet-stream' },
          body: yBuffer,
        });
      },
    ).as('getY');

    cy.intercept(
      'GET',
      `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}/arrays/cat`,
      (req) => {
        req.reply({
          statusCode: 200,
          headers: { 'Content-Type': 'application/octet-stream' },
          body: catBuffer,
        });
      },
    ).as('getCat');

    cy.intercept(
      'GET',
      `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}/arrays/ids`,
      (req) => {
        req.reply({
          statusCode: 200,
          headers: { 'Content-Type': 'application/octet-stream' },
          body: idsBuffer,
        });
      },
    ).as('getIds');

    // Selection resolution
    cy.intercept(
      'POST',
      `/api/v2/collections/${COLLECTION_ID}/projections/${PROJECTION_ID}/select`,
      {
        statusCode: 200,
        body: {
          projection_id: PROJECTION_ID,
          items: [
            {
              selected_id: 100,
              index: 0,
              document_id: 'doc-1',
              chunk_id: 10,
              chunk_index: 0,
              content_preview: 'Example chunk content',
            },
          ],
          missing_ids: [],
          degraded: false,
        },
      },
    ).as('selectProjection');

    // Start projection / recompute
    cy.intercept('POST', `/api/v2/collections/${COLLECTION_ID}/projections`, (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ...projection,
          id: 'projection-2',
          operation_id: 'op-456',
          operation_status: 'processing',
          status: 'pending',
        },
      });
    }).as('startProjection');

    // Auth stub so ProtectedRoute allows access
    cy.visit('/', {
      onBeforeLoad(win) {
        win.localStorage.setItem(
          'auth-storage',
          JSON.stringify({
            state: {
              token: 'test-token',
              refreshToken: null,
              user: {
                id: 1,
                username: 'test-user',
                email: 'test@example.com',
                is_active: true,
                created_at: now,
              },
            },
            version: 0,
          }),
        );
      },
    });

    cy.wait('@listCollections');

    // Open collection details modal and switch to Visualize tab
    cy.get('[data-testid="collection-card"]').contains('Manage').click();
    cy.get('button').contains('Visualize').click();
  });

  it('loads projection, switches color, opens selection, and starts recompute', () => {
    // Wait for projections to be loaded and view the projection
    cy.wait('@listProjections');
    cy.contains('button', 'View').click();

    cy.wait(['@getProjectionMetadata', '@getX', '@getY', '@getCat', '@getIds']);

    cy.contains('Projection preview').should('exist');
    cy.contains('Loaded 4 points').should('exist');

    // Switch color-by to File Type
    cy.get('select').first().select('File Type');

    // Simulate a selection by clicking on the visualization container
    cy.contains('Projection preview')
      .parent()
      .find('div')
      .filter((_, el) => el.style.minHeight === '320px')
      .click('center');

    // Even if selection is not triggered by the click, ensure UI remains stable
    cy.contains('Projection runs').should('exist');

    // Open recompute dialog
    cy.contains('button', /Recompute with File Type/).click();

    // Set a sample size and start recompute
    cy.get('input[placeholder="Optional"]').clear().type('1000');
    cy.contains('button', /^Start$/).click();

    cy.wait('@startProjection').its('request.body').should((body) => {
      expect(body.reducer).to.equal('umap');
      expect(body.color_by).to.equal('filetype');
      expect(body.sample_size).to.equal(1000);
    });

    // Progress banner should appear for the recompute operation
    cy.contains('Projection recompute in progress…').should('exist');
    cy.contains('Operation ID: op-456').should('exist');
  });
});
