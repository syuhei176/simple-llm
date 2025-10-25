import { SimpleLLM } from './llm';
import { trainingData, createVocab } from './training-data';
import { ModelStorage } from './storage';

// グローバル変数
let llm: SimpleLLM;
let isTraining = false;
let isTrainingComplete = false;
const storage = new ModelStorage();

// リポジトリ内のモデルをロード
async function loadModelFromRepo(modelName: string = 'default-latest'): Promise<any | null> {
  try {
    // models/ディレクトリからモデルをフェッチ
    const response = await fetch(`../models/${modelName}.json`);
    if (!response.ok) {
      console.warn(`Model file not found: models/${modelName}.json`);
      return null;
    }
    const modelData = await response.json();
    console.log('Model loaded from repository:', modelName);
    return modelData;
  } catch (error) {
    console.error('Error loading model from repo:', error);
    return null;
  }
}

// 起動時に自動でモデルをロード
async function tryAutoLoadModel() {
  const statusDiv = document.getElementById('status') as HTMLDivElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const predictButton = document.getElementById('predict-button') as HTMLButtonElement;
  const userInput = document.getElementById('user-input') as HTMLInputElement;

  statusDiv.textContent = 'Checking for saved models...';

  // リポジトリ内のモデルをロード
  const modelData = await loadModelFromRepo('default-latest');

  if (modelData) {
    try {
      llm = SimpleLLM.deserialize(modelData);
      isTrainingComplete = true;
      predictButton.disabled = false;
      userInput.disabled = false;

      const message = document.createElement('div');
      message.className = 'message success';
      message.textContent = '✓ Pre-trained model loaded! You can start chatting immediately.';
      outputDiv.appendChild(message);

      if (modelData.metadata) {
        const metaMessage = document.createElement('div');
        metaMessage.className = 'message';
        metaMessage.innerHTML = `
          <strong>Model Info:</strong><br>
          - Name: ${modelData.metadata.name || 'N/A'}<br>
          - Training samples: ${modelData.metadata.trainingSamples || 'N/A'}<br>
          - Vocab size: ${modelData.config.vocabSize || 'N/A'}<br>
          - Created: ${modelData.metadata.createdAt ? new Date(modelData.metadata.createdAt).toLocaleString() : 'N/A'}
        `;
        outputDiv.appendChild(metaMessage);
      }

      statusDiv.textContent = 'Pre-trained model ready!';
    } catch (error) {
      console.error('Error deserializing model:', error);
      statusDiv.textContent = 'Failed to load pre-trained model';
    }
  } else {
    statusDiv.textContent = 'Ready. Click "Train Model" to start or "Load Model" to load a saved model.';
  }
}

// 初期化
async function init() {
  const vocab = createVocab(trainingData);
  const embeddingDim = 64; // より豊かな表現が可能に
  const numLayers = 3; // 複数のTransformerレイヤーを使用
  llm = new SimpleLLM(vocab, embeddingDim, numLayers);

  console.log('Vocabulary:', vocab);
  console.log('Vocabulary size:', vocab.length);
  console.log('First 20 vocab words:', vocab.slice(0, 20));
  console.log('Training data count:', trainingData.length);

  // 起動時にリポジトリ内のモデルを自動ロード
  await tryAutoLoadModel();

  // UI要素の取得
  const trainButton = document.getElementById('train-button') as HTMLButtonElement;
  const predictButton = document.getElementById('predict-button') as HTMLButtonElement;
  const saveButton = document.getElementById('save-button') as HTMLButtonElement;
  const loadButton = document.getElementById('load-button') as HTMLButtonElement;
  const userInput = document.getElementById('user-input') as HTMLInputElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const statusDiv = document.getElementById('status') as HTMLDivElement;

  // トレーニングボタンのイベント
  trainButton.addEventListener('click', async () => {
    if (isTraining) return;

    isTraining = true;
    isTrainingComplete = false;
    trainButton.disabled = true;
    predictButton.disabled = true;
    userInput.disabled = true;
    statusDiv.textContent = 'Training...';
    outputDiv.innerHTML = '<div class="message">Training started...</div>';

    // 非同期でトレーニングを実行（UIをブロックしないように）
    setTimeout(() => {
      try {
        const epochs = 100;
        llm.train(trainingData, epochs);
        statusDiv.textContent = 'Training completed!';
        outputDiv.innerHTML += '<div class="message success">Training completed! You can now chat with the AI.</div>';
        isTrainingComplete = true;
        predictButton.disabled = false;
        userInput.disabled = false;
      } catch (error) {
        statusDiv.textContent = 'Training failed';
        outputDiv.innerHTML += `<div class="message error">Error: ${error}</div>`;
        console.error(error);
      } finally {
        isTraining = false;
        trainButton.disabled = false;
      }
    }, 100);
  });

  // 予測ボタンのイベント
  predictButton.addEventListener('click', () => {
    const input = userInput.value.trim();
    if (!input) {
      alert('Please enter some text');
      return;
    }

    if (!isTrainingComplete) {
      alert('Please train the model first');
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

  // 保存ボタンのイベント
  saveButton.addEventListener('click', async () => {
    if (!isTrainingComplete) {
      alert('Please train the model first before saving');
      return;
    }

    try {
      saveButton.disabled = true;
      statusDiv.textContent = 'Saving model...';

      const modelData = llm.serialize();
      await storage.saveModel(modelData);

      const message = document.createElement('div');
      message.className = 'message success';
      message.textContent = 'Model saved successfully to IndexedDB!';
      outputDiv.appendChild(message);
      outputDiv.scrollTop = outputDiv.scrollHeight;

      statusDiv.textContent = 'Model saved!';
    } catch (error) {
      const errorMessage = document.createElement('div');
      errorMessage.className = 'message error';
      errorMessage.textContent = `Failed to save model: ${error}`;
      outputDiv.appendChild(errorMessage);
      console.error(error);
      statusDiv.textContent = 'Save failed';
    } finally {
      saveButton.disabled = false;
    }
  });

  // 読み込みボタンのイベント
  loadButton.addEventListener('click', async () => {
    try {
      loadButton.disabled = true;
      statusDiv.textContent = 'Loading model...';

      const modelData = await storage.loadModel();

      if (modelData) {
        llm = SimpleLLM.deserialize(modelData);
        isTrainingComplete = true;
        predictButton.disabled = false;
        userInput.disabled = false;

        const message = document.createElement('div');
        message.className = 'message success';
        message.textContent = 'Model loaded successfully from IndexedDB! You can now chat with the AI.';
        outputDiv.appendChild(message);
        outputDiv.scrollTop = outputDiv.scrollHeight;

        statusDiv.textContent = 'Model loaded!';
      } else {
        const message = document.createElement('div');
        message.className = 'message error';
        message.textContent = 'No saved model found. Please train a model first.';
        outputDiv.appendChild(message);
        statusDiv.textContent = 'No saved model found';
      }
    } catch (error) {
      const errorMessage = document.createElement('div');
      errorMessage.className = 'message error';
      errorMessage.textContent = `Failed to load model: ${error}`;
      outputDiv.appendChild(errorMessage);
      console.error(error);
      statusDiv.textContent = 'Load failed';
    } finally {
      loadButton.disabled = false;
    }
  });
}

// ページ読み込み時に初期化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
