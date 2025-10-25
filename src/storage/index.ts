// IndexedDBを使用したモデルストレージ
export class ModelStorage {
  private dbName = 'SimpleLLMDB';
  private storeName = 'models';
  private version = 2; // バイナリサポートのためバージョンアップ

  // IndexedDBを開く
  private async openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'id' });
        }
      };
    });
  }

  // モデルを保存（バイナリ形式）
  async saveModel(modelData: Uint8Array | any, modelId: string = 'default', isBinary: boolean = true): Promise<void> {
    const db = await this.openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      const data = {
        id: modelId,
        modelData,
        timestamp: Date.now(),
        isBinary, // フォーマットを識別
      };

      const request = store.put(data);

      request.onsuccess = () => {
        console.log(`Model saved successfully to IndexedDB (${isBinary ? 'binary' : 'JSON'} format)`);
        resolve();
      };
      request.onerror = () => reject(request.error);

      transaction.oncomplete = () => db.close();
    });
  }

  // モデルを読み込み
  async loadModel(modelId: string = 'default'): Promise<{ data: any; isBinary: boolean } | null> {
    const db = await this.openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(modelId);

      request.onsuccess = () => {
        const result = request.result;
        if (result) {
          console.log(`Model loaded successfully from IndexedDB (${result.isBinary ? 'binary' : 'JSON'} format)`);
          resolve({ data: result.modelData, isBinary: result.isBinary || false });
        } else {
          console.log('No saved model found');
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);

      transaction.oncomplete = () => db.close();
    });
  }

  // 保存されているモデルの一覧を取得
  async listModels(): Promise<{ id: string; timestamp: number }[]> {
    const db = await this.openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.getAllKeys();

      request.onsuccess = () => {
        const keys = request.result as string[];
        // 各キーのタイムスタンプも取得
        const getAllRequest = store.getAll();
        getAllRequest.onsuccess = () => {
          const models = getAllRequest.result.map((item: any) => ({
            id: item.id,
            timestamp: item.timestamp,
          }));
          resolve(models);
        };
        getAllRequest.onerror = () => reject(getAllRequest.error);
      };
      request.onerror = () => reject(request.error);

      transaction.oncomplete = () => db.close();
    });
  }

  // モデルを削除
  async deleteModel(modelId: string = 'default'): Promise<void> {
    const db = await this.openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(modelId);

      request.onsuccess = () => {
        console.log('Model deleted successfully from IndexedDB');
        resolve();
      };
      request.onerror = () => reject(request.error);

      transaction.oncomplete = () => db.close();
    });
  }
}
