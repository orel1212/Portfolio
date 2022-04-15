import { TestBed } from "@angular/core/testing";
import { languageConverter } from "./langConverter.pipe";

describe('langConverter: Pipe', () => {
    beforeEach(() => {
      TestBed.configureTestingModule({
          declarations:[languageConverter],
          imports: [  ],
          providers: []
      });
    });
  
    it('check if lang pipe is working as expected', () => {
      let langConverter= new languageConverter();
      expect(langConverter.transform("he")).toBe("Hebrew");
  
    });
});