export class Recipe {
  constructor(
    public title: string,
    public url: string,
    public imgsrc: string,
    public userid?: number,
    public _id?: number,
    public updatedAt?: Date,
    public createdAt?: Date,
    public lastUpdatedBy?: string,
  ) { }
}