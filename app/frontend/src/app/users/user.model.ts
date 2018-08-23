export class User {
  constructor(
    public username: string,
    public password: string,
    public name?: string,
    public _id?: number,
    public updatedAt?: Date,
    public createdAt?: Date,
    public lastUpdatedBy?: string,
  ) { }
}