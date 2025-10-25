import { SimpleLLM } from './llm';

// グローバル変数
let llm: SimpleLLM;
let isModelReady = false;

// リポジトリ内のモデルをロード
async function loadModelFromRepo(modelName: string = 'default-latest'): Promise<SimpleLLM | null> {
  try {
    // まずバイナリフォーマット(.msgpack)を試す
    let response = await fetch(`./models/${modelName}.msgpack`);
    if (response.ok) {
      const arrayBuffer = await response.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      const model = SimpleLLM.deserializeBinary(uint8Array);
      console.log('Model loaded from repository (binary format):', modelName);
      return model;
    }

    // バイナリがなければJSON形式を試す（後方互換性）
    console.log('Binary format not found, trying JSON...');
    response = await fetch(`./models/${modelName}.json`);
    if (!response.ok) {
      console.warn(`Model file not found: models/${modelName}.json or .msgpack`);
      return null;
    }
    const modelData = await response.json();
    console.log('Model loaded from repository (JSON format):', modelName);
    const model = SimpleLLM.deserialize(modelData);
    return model;
  } catch (error) {
    console.error('Error loading model from repo:', error);
    return null;
  }
}

// 起動時に自動でモデルをロード
async function loadModel(modelName: string = 'default-latest') {
  const statusDiv = document.getElementById('status') as HTMLDivElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const predictButton = document.getElementById('predict-button') as HTMLButtonElement;
  const userInput = document.getElementById('user-input') as HTMLInputElement;

  statusDiv.textContent = `Loading model from repository: ${modelName}...`;

  // チャット履歴をクリア
  outputDiv.innerHTML = '';

  // リポジトリ内のモデルをロード
  const model = await loadModelFromRepo(modelName);

  if (model) {
    try {
      llm = model;
      isModelReady = true;
      predictButton.disabled = false;
      userInput.disabled = false;

      const message = document.createElement('div');
      message.className = 'message success';
      message.textContent = '✓ Model loaded successfully! You can start chatting now.';
      outputDiv.appendChild(message);

      // モデル情報を表示
      const metaMessage = document.createElement('div');
      metaMessage.className = 'message';
      metaMessage.innerHTML = `
        <strong>Model Info:</strong><br>
        - Vocab size: ${llm.vocabSize}<br>
        - Embedding dimension: ${llm.embeddingDim}<br>
        - Number of layers: ${llm.numLayers}<br>
        - Number of heads: ${llm.numHeads}
      `;
      outputDiv.appendChild(metaMessage);

      statusDiv.textContent = 'Model ready! Start chatting below.';
    } catch (error) {
      console.error('Error loading model:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'message error';
      errorMessage.textContent = '✗ Failed to load model. Please check the console for details.';
      outputDiv.appendChild(errorMessage);
      statusDiv.textContent = 'Failed to load model';
    }
  } else {
    const errorMessage = document.createElement('div');
    errorMessage.className = 'message error';
    errorMessage.textContent = '✗ Model file not found. Please ensure default-latest.msgpack or .json exists in the models directory.';
    outputDiv.appendChild(errorMessage);
    statusDiv.textContent = 'Model not found';
  }
}

// 初期化
async function init() {
  // 起動時にリポジトリ内のモデルを自動ロード
  await loadModel();

  // UI要素の取得
  const predictButton = document.getElementById('predict-button') as HTMLButtonElement;
  const userInput = document.getElementById('user-input') as HTMLInputElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const loadModelButton = document.getElementById('load-model-button') as HTMLButtonElement;

  // 予測ボタンのイベント
  predictButton.addEventListener('click', () => {
    const input = userInput.value.trim();
    if (!input) {
      alert('Please enter some text');
      return;
    }

    if (!isModelReady) {
      alert('Model is not ready yet. Please wait for it to load.');
      return;
    }

    try {
      const response = llm.predict(input, 10);
      const userMessage = document.createElement('div');
      userMessage.className = 'message user-message';
      userMessage.textContent = `You: ${input}`;
      outputDiv.appendChild(userMessage);

      const aiMessage = document.createElement('div');
      aiMessage.className = 'message ai-message';
      aiMessage.textContent = `AI: ${response}`;
      outputDiv.appendChild(aiMessage);

      outputDiv.scrollTop = outputDiv.scrollHeight;
      userInput.value = '';
    } catch (error) {
      const errorMessage = document.createElement('div');
      errorMessage.className = 'message error';
      errorMessage.textContent = `Error: ${error}`;
      outputDiv.appendChild(errorMessage);
      console.error(error);
    }
  });

  // Enterキーで送信
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      predictButton.click();
    }
  });

  // モデル読み込みボタンのイベント
  loadModelButton.addEventListener('click', async () => {
    const selectedModel = modelSelect.value;
    predictButton.disabled = true;
    userInput.disabled = true;
    loadModelButton.disabled = true;

    await loadModel(selectedModel);

    loadModelButton.disabled = false;
  });
}

// ページ読み込み時に初期化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
