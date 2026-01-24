import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<{ id: string; username: string; password: string } | undefined>;
  getUserByUsername(username: string): Promise<{ id: string; username: string; password: string } | undefined>;
  createUser(user: { username: string; password: string }): Promise<{ id: string; username: string; password: string }>;
}

export class MemStorage implements IStorage {
  private users: Map<string, { id: string; username: string; password: string }>;

  constructor() {
    this.users = new Map();
  }

  async getUser(id: string): Promise<{ id: string; username: string; password: string } | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<{ id: string; username: string; password: string } | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: { username: string; password: string }): Promise<{ id: string; username: string; password: string }> {
    const id = randomUUID();
    const user = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
}

export const storage = new MemStorage();
