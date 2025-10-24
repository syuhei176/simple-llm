import { SimpleLLM } from './llm';
import { trainingData, createVocab } from './training-data';

// グローバル変数
let llm: SimpleLLM;
let isTraining = false;
let isTrainingComplete = false;

// 初期化
function init() {
  const vocab = createVocab(trainingData);
  const embeddingDim = 64; // より豊かな表現が可能に
  const numLayers = 3; // 複数のTransformerレイヤーを使用
  llm = new SimpleLLM(vocab, embeddingDim, numLayers);

  console.log('Vocabulary:', vocab);
  console.log('Vocabulary size:', vocab.length);
  console.log('First 20 vocab words:', vocab.slice(0, 20));
  console.log('Training data count:', trainingData.length);

  // UI要素の取得
  const trainButton = document.getElementById('train-button') as HTMLButtonElement;
  const predictButton = document.getElementById('predict-button') as HTMLButtonElement;
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

  statusDiv.textContent = 'Ready. Click "Train Model" to start.';
}

// ページ読み込み時に初期化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
