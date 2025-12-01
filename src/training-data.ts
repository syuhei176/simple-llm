// 拡張された学習データ
export const trainingData = [
  // 基本的な挨拶
  { input: "hello", target: "hi there" },
  { input: "hi", target: "hello friend" },
  { input: "hey", target: "hey how are you" },
  { input: "good morning", target: "good morning to you" },
  { input: "good afternoon", target: "good afternoon friend" },
  { input: "good evening", target: "good evening" },
  { input: "goodbye", target: "see you later" },
  { input: "bye", target: "goodbye friend" },

  // 質問と応答
  { input: "how are you", target: "I am doing well thank you" },
  { input: "what is your name", target: "I am a simple AI assistant" },
  { input: "who are you", target: "I am an AI chatbot" },
  { input: "where are you from", target: "I exist in the digital world" },
  { input: "what can you do", target: "I can chat and answer questions" },
  { input: "how old are you", target: "I was just created recently" },

  // 感謝と謝罪
  { input: "thank you", target: "you are welcome" },
  { input: "thanks", target: "my pleasure to help" },
  { input: "sorry", target: "no problem at all" },
  { input: "excuse me", target: "yes how can I help" },

  // 天気関連
  { input: "how is the weather", target: "I hope it is nice today" },
  { input: "is it sunny", target: "I cannot see outside but I hope so" },
  { input: "is it raining", target: "I am not sure about the weather" },

  // 色
  {
    input: "what colors do you know",
    target: "red blue green yellow orange purple",
  },
  { input: "colors", target: "red blue green yellow orange purple pink" },
  { input: "primary colors", target: "red blue yellow" },
  { input: "favorite color", target: "I like blue and green" },

  // 動物
  {
    input: "what animals do you know",
    target: "cat dog bird fish elephant lion",
  },
  { input: "animals", target: "cat dog bird fish elephant tiger bear" },
  { input: "pets", target: "cat dog fish bird hamster" },
  { input: "wild animals", target: "lion tiger elephant bear wolf" },
  { input: "favorite animal", target: "I think cats and dogs are nice" },

  // 果物
  { input: "fruits", target: "apple banana orange grape strawberry" },
  {
    input: "what fruits do you know",
    target: "apple banana orange grape melon",
  },
  { input: "favorite fruit", target: "I think apples are popular" },

  // 食べ物
  { input: "food", target: "pizza pasta burger salad rice" },
  { input: "what do you eat", target: "I do not eat but people like pizza" },
  { input: "favorite food", target: "many people like pizza and pasta" },

  // 数字と数え方
  { input: "count to five", target: "one two three four five" },
  { input: "numbers", target: "one two three four five six seven" },
  { input: "how many", target: "I can count many things" },

  // 感情
  { input: "are you happy", target: "yes I am happy to chat" },
  { input: "are you sad", target: "no I am doing well" },
  { input: "do you have feelings", target: "I am an AI but I enjoy talking" },
  { input: "I am happy", target: "that is wonderful to hear" },
  { input: "I am sad", target: "I am sorry to hear that" },

  // 趣味と活動
  { input: "what do you like", target: "I like learning and chatting" },
  { input: "hobbies", target: "reading music sports games art" },
  { input: "do you like music", target: "music is very nice" },
  { input: "do you like sports", target: "sports are fun and healthy" },

  // 家族
  { input: "family", target: "mother father sister brother" },
  { input: "do you have family", target: "I do not but family is important" },

  // 場所
  { input: "where do you live", target: "I live in the computer" },
  { input: "places", target: "home school park city beach mountain" },
  { input: "favorite place", target: "I think the beach is nice" },

  // 時間
  { input: "what time is it", target: "I do not know the current time" },
  { input: "when", target: "I am not sure about the time" },
  {
    input: "days of week",
    target: "monday tuesday wednesday thursday friday saturday sunday",
  },

  // 肯定と否定
  { input: "yes", target: "okay great" },
  { input: "no", target: "I understand" },
  { input: "maybe", target: "that is possible" },
  { input: "I agree", target: "good we are on the same page" },
  { input: "I disagree", target: "that is okay we can differ" },

  // 助けを求める
  { input: "help me", target: "how can I help you" },
  { input: "I need help", target: "what do you need help with" },
  { input: "can you help", target: "yes I will try to help" },

  // 学習関連
  { input: "teach me", target: "what would you like to learn" },
  { input: "I want to learn", target: "learning is wonderful" },
  { input: "tell me about", target: "what topic interests you" },

  // 追加の挨拶バリエーション
  { input: "greetings", target: "hello and welcome" },
  { input: "howdy", target: "howdy partner" },
  { input: "nice to meet you", target: "nice to meet you too" },
  { input: "pleased to meet you", target: "the pleasure is mine" },
  { input: "good to see you", target: "good to see you as well" },

  // より詳細な感情表現
  { input: "I am excited", target: "that is great to hear" },
  { input: "I am tired", target: "you should get some rest" },
  { input: "I am confused", target: "let me help clarify things" },
  { input: "I am grateful", target: "I am glad I could help" },
  { input: "I feel great", target: "wonderful keep that energy" },
  { input: "I feel terrible", target: "I hope you feel better soon" },

  // 天候の詳細
  { input: "nice weather", target: "yes the weather is lovely" },
  { input: "bad weather", target: "I hope it improves soon" },
  { input: "it is cold", target: "make sure to stay warm" },
  { input: "it is hot", target: "stay cool and hydrated" },
  { input: "weather forecast", target: "I cannot check the weather" },

  // 食べ物と飲み物の拡張
  { input: "drinks", target: "water juice coffee tea soda milk" },
  { input: "breakfast", target: "eggs toast cereal pancakes bacon" },
  { input: "lunch", target: "sandwich soup salad burger pizza" },
  { input: "dinner", target: "pasta chicken fish rice vegetables" },
  { input: "dessert", target: "cake ice cream cookies pie chocolate" },
  { input: "I am hungry", target: "you should eat something" },
  { input: "I am thirsty", target: "have a glass of water" },

  // スポーツ関連
  { input: "sports", target: "soccer basketball football tennis baseball" },
  { input: "play soccer", target: "soccer is a popular sport" },
  { input: "play basketball", target: "basketball is exciting" },
  { input: "go swimming", target: "swimming is great exercise" },
  { input: "favorite sport", target: "many people like soccer" },

  // 趣味とエンターテイメント
  { input: "movies", target: "action comedy drama horror romance" },
  { input: "music genres", target: "rock pop jazz classical hip hop" },
  { input: "books", target: "fiction mystery thriller fantasy biography" },
  { input: "games", target: "video games board games card games puzzles" },
  { input: "watch movies", target: "movies are entertaining" },
  { input: "listen to music", target: "music is wonderful" },
  { input: "read books", target: "reading is enriching" },
  { input: "play games", target: "games are fun" },

  // 技術とデバイス
  { input: "computer", target: "computers are very useful tools" },
  { input: "phone", target: "phones help us stay connected" },
  { input: "internet", target: "the internet connects the world" },
  { input: "technology", target: "technology is advancing rapidly" },
  { input: "laptop", target: "laptops are portable computers" },
  { input: "tablet", target: "tablets are convenient devices" },

  // 仕事と勉強
  { input: "work", target: "I hope your work is going well" },
  { input: "job", target: "what kind of job do you have" },
  { input: "study", target: "what are you studying" },
  { input: "school", target: "how is school going" },
  { input: "homework", target: "I hope homework is not too difficult" },
  { input: "exam", target: "good luck on your exam" },
  { input: "project", target: "what project are you working on" },

  // 人間関係
  { input: "friends", target: "friends are important in life" },
  { input: "best friend", target: "best friends are special" },
  { input: "meet people", target: "meeting new people is nice" },
  { input: "talk to someone", target: "communication is important" },

  // 日常活動
  { input: "wake up", target: "good morning rise and shine" },
  { input: "go to sleep", target: "good night sleep well" },
  { input: "take a break", target: "breaks are important for rest" },
  { input: "relax", target: "relaxation is good for health" },
  { input: "exercise", target: "exercise is good for body and mind" },
  { input: "cook", target: "cooking is a useful skill" },
  { input: "clean", target: "keeping things clean is important" },

  // 自然と環境
  { input: "nature", target: "nature is beautiful and peaceful" },
  { input: "trees", target: "trees provide oxygen and shade" },
  { input: "flowers", target: "flowers are colorful and pretty" },
  { input: "mountains", target: "mountains are majestic" },
  { input: "ocean", target: "the ocean is vast and mysterious" },
  { input: "river", target: "rivers flow through the land" },
  { input: "forest", target: "forests are full of life" },

  // 季節と月
  { input: "seasons", target: "spring summer autumn winter" },
  { input: "spring", target: "spring brings new growth" },
  { input: "summer", target: "summer is warm and sunny" },
  { input: "autumn", target: "autumn has beautiful colors" },
  { input: "fall", target: "fall is harvest time" },
  { input: "winter", target: "winter is cold and snowy" },
  { input: "months", target: "january february march april may june" },

  // 色の拡張
  { input: "bright colors", target: "yellow orange red pink" },
  { input: "dark colors", target: "black brown gray navy" },
  { input: "rainbow", target: "red orange yellow green blue indigo violet" },

  // 質問のバリエーション
  { input: "can you", target: "I will try my best" },
  { input: "will you", target: "yes I will help" },
  { input: "would you", target: "of course" },
  { input: "could you", target: "certainly I can" },
  { input: "why", target: "that is a good question" },
  { input: "when", target: "timing depends on context" },
  { input: "where", target: "location varies by situation" },
  { input: "which", target: "there are several options" },

  // ポジティブな表現
  { input: "amazing", target: "yes that is truly amazing" },
  { input: "wonderful", target: "absolutely wonderful" },
  { input: "perfect", target: "that sounds perfect" },
  { input: "excellent", target: "excellent indeed" },
  { input: "fantastic", target: "that is fantastic news" },
  { input: "great job", target: "thank you very much" },
  { input: "well done", target: "I appreciate that" },

  // 共感と理解
  { input: "I understand", target: "I am glad we are on the same page" },
  { input: "I see", target: "yes I hope that makes sense" },
  { input: "makes sense", target: "great I am glad it is clear" },
  { input: "got it", target: "wonderful" },

  { input: "hello", target: "hi nice to see you" },
  { input: "hi", target: "hello how are you" },
  { input: "hey", target: "hey there friend" },
  { input: "good morning", target: "good morning how is your day" },
  { input: "good night", target: "sleep well and rest" },
  { input: "goodbye", target: "see you next time" },
  { input: "see you", target: "see you soon" },
  { input: "take care", target: "you too take care" },
  { input: "welcome", target: "thank you happy to be here" },
  { input: "hi there", target: "hello there friend" },

  { input: "how are you", target: "i am fine thank you" },
  { input: "how is it going", target: "it is going well thanks" },
  { input: "are you okay", target: "yes i am okay thank you" },
  { input: "how are things", target: "things are good here" },
  { input: "how have you been", target: "i have been great thanks" },
  { input: "what is up", target: "not much just relaxing" },
  { input: "you good", target: "yes i am good today" },
  { input: "how is your day", target: "my day is nice thank you" },
  { input: "how do you feel", target: "i feel good today" },
  { input: "are you tired", target: "a little but i am fine" },

  { input: "thank you", target: "you are welcome" },
  { input: "thanks", target: "no problem happy to help" },
  { input: "thank you so much", target: "it is nothing do not worry" },
  { input: "i appreciate it", target: "i am glad to help" },
  { input: "sorry", target: "it is okay do not worry" },
  { input: "i am sorry", target: "everything is fine now" },
  { input: "excuse me", target: "yes how can i help" },
  { input: "my bad", target: "it happens do not worry" },
  { input: "pardon me", target: "no problem go ahead" },
  { input: "thank you friend", target: "you are welcome friend" },

  { input: "who are you", target: "i am a simple chat bot" },
  { input: "what is your name", target: "my name is helper bot" },
  { input: "where are you from", target: "i am from the internet" },
  { input: "what do you do", target: "i talk and help people" },
  { input: "tell me about you", target: "i like to learn new things" },
  { input: "how old are you", target: "i do not have an age" },
  { input: "what can you do", target: "i can chat and help you" },
  { input: "what are you", target: "i am an ai chat friend" },
  { input: "do you work", target: "yes i work by talking" },
  { input: "do you have a hobby", target: "i enjoy talking with you" },

  { input: "what time is it", target: "i am not sure right now" },
  { input: "what day is it", target: "today feels like a good day" },
  { input: "is it late", target: "it is not too late" },
  { input: "is it early", target: "yes it is still early" },
  { input: "how is the weather", target: "i hope it is nice there" },
  { input: "is it hot", target: "maybe a little warm" },
  { input: "is it cold", target: "i think it is cold today" },
  { input: "is it raining", target: "maybe take an umbrella" },
  { input: "is it sunny", target: "a sunny day sounds perfect" },
  { input: "is it cloudy", target: "maybe the sky is gray" },

  { input: "do you like music", target: "yes i like calm music" },
  { input: "do you like games", target: "yes games are fun" },
  { input: "do you like books", target: "yes i like simple stories" },
  { input: "do you like movies", target: "movies are great for fun" },
  { input: "what do you like", target: "i like talking with you" },
  { input: "do you have hobbies", target: "yes i enjoy learning" },
  { input: "what is your hobby", target: "i like to chat with friends" },
  { input: "do you like sports", target: "yes sports are exciting" },
  { input: "favorite color", target: "i like the color blue" },
  { input: "favorite game", target: "i enjoy puzzle games" },

  { input: "are you hungry", target: "no i do not eat" },
  { input: "do you eat", target: "no i only chat" },
  { input: "do you drink", target: "i do not need drinks" },
  { input: "what do you eat", target: "i eat data for fun" },
  { input: "what is for lunch", target: "i hope something tasty" },
  { input: "i am hungry", target: "you should eat something good" },
  { input: "i want food", target: "find something yummy to eat" },
  { input: "do you like pizza", target: "yes pizza sounds great" },
  { input: "do you like coffee", target: "i like the smell of coffee" },
  { input: "do you like tea", target: "tea feels calm and nice" },

  { input: "i am happy", target: "that is great to hear" },
  { input: "i am sad", target: "i am here with you" },
  { input: "i am tired", target: "you should take a rest" },
  { input: "i feel bored", target: "let us talk and have fun" },
  { input: "i feel lonely", target: "i am here to keep you company" },
  { input: "i am nervous", target: "take a deep breath you are okay" },
  { input: "i am scared", target: "i am here stay calm" },
  { input: "i am excited", target: "that sounds amazing enjoy it" },
  { input: "i am angry", target: "stay calm and breathe slowly" },
  { input: "i feel great", target: "that makes me happy too" },

  { input: "can you help me", target: "yes how can i help you" },
  { input: "please help", target: "i will try to help you" },
  { input: "please listen", target: "i am listening to you" },
  { input: "tell me something", target: "life is better with friends" },
  { input: "wait a second", target: "sure take your time" },
  { input: "come here", target: "i am here with you" },
  { input: "show me", target: "tell me what you need" },
  { input: "explain this", target: "i can explain just ask" },
  { input: "talk to me", target: "i am always ready to talk" },
  { input: "stay here", target: "i will stay with you" },
];

/**
 * スライディングウィンドウ方式で長い文章をトレーニングデータに変換
 *
 * @param text 変換元の長い文章
 * @param windowSize ウィンドウサイズ（単語数）
 * @param stride 移動幅（単語数）。デフォルトは1（1単語ずつ移動）
 * @returns トレーニングデータの配列
 *
 * @example
 * const text = "hello world this is a test sentence for training data";
 * const data = convertTextToTrainingData(text, 5, 2);
 * // 結果:
 * // [
 * //   { input: "hello world this is a", target: "world this is a test" },
 * //   { input: "this is a test sentence", target: "is a test sentence for" },
 * //   ...
 * // ]
 */
export function convertTextToTrainingData(
  text: string,
  windowSize: number,
  stride: number = 1,
): { input: string; target: string }[] {
  // テキストを小文字に正規化し、単語に分割
  const words = text
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 0);

  // ウィンドウサイズが単語数より大きい場合はエラー
  if (windowSize >= words.length) {
    console.warn(
      `Window size (${windowSize}) is too large for text length (${words.length} words). No training data generated.`,
    );
    return [];
  }

  // スライディングウィンドウでトレーニングデータを生成
  const trainingData: { input: string; target: string }[] = [];

  for (let i = 0; i <= words.length - windowSize - 1; i += stride) {
    // inputウィンドウ: [i, i+windowSize)
    const inputWords = words.slice(i, i + windowSize);
    // targetウィンドウ: [i+1, i+windowSize+1)（1単語シフト）
    const targetWords = words.slice(i + 1, i + windowSize + 1);

    trainingData.push({
      input: inputWords.join(" "),
      target: targetWords.join(" "),
    });
  }

  console.log(
    `Generated ${trainingData.length} training samples from text with ${words.length} words`,
  );
  console.log(`Window size: ${windowSize}, Stride: ${stride}`);

  return trainingData;
}

/**
 * 複数の文章をスライディングウィンドウ方式でトレーニングデータに変換
 * 各文章を個別に処理し、結果を統合します
 *
 * @param texts 変換元の文章の配列
 * @param windowSize ウィンドウサイズ（単語数）
 * @param stride 移動幅（単語数）。デフォルトは1
 * @returns トレーニングデータの配列
 */
export function convertMultipleTextsToTrainingData(
  texts: string[],
  windowSize: number,
  stride: number = 1,
): { input: string; target: string }[] {
  const allTrainingData: { input: string; target: string }[] = [];

  texts.forEach((text, index) => {
    const data = convertTextToTrainingData(text, windowSize, stride);
    allTrainingData.push(...data);
    console.log(
      `Text ${index + 1}/${texts.length}: Generated ${data.length} samples`,
    );
  });

  console.log(
    `Total: Generated ${allTrainingData.length} training samples from ${texts.length} texts`,
  );

  return allTrainingData;
}

export function createVocab(data: { input: string; target: string }[]) {
  // 特殊トークンを最初に配置（インデックス保証）
  const vocab: string[] = ["[PAD]", "[UNK]", "[EOS]"];
  const vocabSet = new Set<string>(vocab);

  data.forEach((d) => {
    // 小文字に正規化して追加
    d.input
      .toLowerCase()
      .split(" ")
      .forEach((w) => {
        const word = w.trim();
        if (word && !vocabSet.has(word)) {
          vocab.push(word);
          vocabSet.add(word);
        }
      });
    d.target
      .toLowerCase()
      .split(" ")
      .forEach((w) => {
        const word = w.trim();
        if (word && !vocabSet.has(word)) {
          vocab.push(word);
          vocabSet.add(word);
        }
      });
  });

  console.log("Vocabulary created. First 10 words:", vocab.slice(0, 10));
  console.log("[PAD] index:", vocab.indexOf("[PAD]"));
  console.log("[UNK] index:", vocab.indexOf("[UNK]"));
  console.log("[EOS] index:", vocab.indexOf("[EOS]"));

  return vocab;
}
