// 拡張された学習データ
export const trainingData = [
  // 基本的な挨拶
  { input: 'hello', target: 'hi there' },
  { input: 'hi', target: 'hello friend' },
  { input: 'hey', target: 'hey how are you' },
  { input: 'good morning', target: 'good morning to you' },
  { input: 'good afternoon', target: 'good afternoon friend' },
  { input: 'good evening', target: 'good evening' },
  { input: 'goodbye', target: 'see you later' },
  { input: 'bye', target: 'goodbye friend' },

  // 質問と応答
  { input: 'how are you', target: 'I am doing well thank you' },
  { input: 'what is your name', target: 'I am a simple AI assistant' },
  { input: 'who are you', target: 'I am an AI chatbot' },
  { input: 'where are you from', target: 'I exist in the digital world' },
  { input: 'what can you do', target: 'I can chat and answer questions' },
  { input: 'how old are you', target: 'I was just created recently' },

  // 感謝と謝罪
  { input: 'thank you', target: 'you are welcome' },
  { input: 'thanks', target: 'my pleasure to help' },
  { input: 'sorry', target: 'no problem at all' },
  { input: 'excuse me', target: 'yes how can I help' },

  // 天気関連
  { input: 'how is the weather', target: 'I hope it is nice today' },
  { input: 'is it sunny', target: 'I cannot see outside but I hope so' },
  { input: 'is it raining', target: 'I am not sure about the weather' },

  // 色
  { input: 'what colors do you know', target: 'red blue green yellow orange purple' },
  { input: 'colors', target: 'red blue green yellow orange purple pink' },
  { input: 'primary colors', target: 'red blue yellow' },
  { input: 'favorite color', target: 'I like blue and green' },

  // 動物
  { input: 'what animals do you know', target: 'cat dog bird fish elephant lion' },
  { input: 'animals', target: 'cat dog bird fish elephant tiger bear' },
  { input: 'pets', target: 'cat dog fish bird hamster' },
  { input: 'wild animals', target: 'lion tiger elephant bear wolf' },
  { input: 'favorite animal', target: 'I think cats and dogs are nice' },

  // 果物
  { input: 'fruits', target: 'apple banana orange grape strawberry' },
  { input: 'what fruits do you know', target: 'apple banana orange grape melon' },
  { input: 'favorite fruit', target: 'I think apples are popular' },

  // 食べ物
  { input: 'food', target: 'pizza pasta burger salad rice' },
  { input: 'what do you eat', target: 'I do not eat but people like pizza' },
  { input: 'favorite food', target: 'many people like pizza and pasta' },

  // 数字と数え方
  { input: 'count to five', target: 'one two three four five' },
  { input: 'numbers', target: 'one two three four five six seven' },
  { input: 'how many', target: 'I can count many things' },

  // 感情
  { input: 'are you happy', target: 'yes I am happy to chat' },
  { input: 'are you sad', target: 'no I am doing well' },
  { input: 'do you have feelings', target: 'I am an AI but I enjoy talking' },
  { input: 'I am happy', target: 'that is wonderful to hear' },
  { input: 'I am sad', target: 'I am sorry to hear that' },

  // 趣味と活動
  { input: 'what do you like', target: 'I like learning and chatting' },
  { input: 'hobbies', target: 'reading music sports games art' },
  { input: 'do you like music', target: 'music is very nice' },
  { input: 'do you like sports', target: 'sports are fun and healthy' },

  // 家族
  { input: 'family', target: 'mother father sister brother' },
  { input: 'do you have family', target: 'I do not but family is important' },

  // 場所
  { input: 'where do you live', target: 'I live in the computer' },
  { input: 'places', target: 'home school park city beach mountain' },
  { input: 'favorite place', target: 'I think the beach is nice' },

  // 時間
  { input: 'what time is it', target: 'I do not know the current time' },
  { input: 'when', target: 'I am not sure about the time' },
  { input: 'days of week', target: 'monday tuesday wednesday thursday friday saturday sunday' },

  // 肯定と否定
  { input: 'yes', target: 'okay great' },
  { input: 'no', target: 'I understand' },
  { input: 'maybe', target: 'that is possible' },
  { input: 'I agree', target: 'good we are on the same page' },
  { input: 'I disagree', target: 'that is okay we can differ' },

  // 助けを求める
  { input: 'help me', target: 'how can I help you' },
  { input: 'I need help', target: 'what do you need help with' },
  { input: 'can you help', target: 'yes I will try to help' },

  // 学習関連
  { input: 'teach me', target: 'what would you like to learn' },
  { input: 'I want to learn', target: 'learning is wonderful' },
  { input: 'tell me about', target: 'what topic interests you' },
];

export function createVocab(data: { input: string, target: string }[]) {
  const vocab = new Set<string>();

  // 特殊トークンを最初に追加
  vocab.add('[PAD]');  // パディング用
  vocab.add('[UNK]');  // 未知語用
  vocab.add('[EOS]');  // 文末用

  data.forEach(d => {
    // 小文字に正規化して追加
    d.input.toLowerCase().split(' ').forEach(w => {
      if (w.trim()) vocab.add(w.trim());
    });
    d.target.toLowerCase().split(' ').forEach(w => {
      if (w.trim()) vocab.add(w.trim());
    });
  });

  return Array.from(vocab);
}
