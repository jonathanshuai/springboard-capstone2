export class Restriction {
  constructor(
    public vegan: boolean,
    public vegetarian: boolean,
    public peanut_free: boolean,
    public userid?: number,
    public _id?: number,
    public updatedAt?: Date,
    public createdAt?: Date,
    public lastUpdatedBy?: string,
  ) { }
}