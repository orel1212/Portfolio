import { TestBed } from "@angular/core/testing";
import { countryConverter } from "./countryConverter.pipe";


describe('countryConverter: Pipe', () => {
    beforeEach(() => {
      TestBed.configureTestingModule({
          declarations:[countryConverter],
          imports: [  ],
          providers: []
      });
    });
  
    it('check if country pipe is working as expected', () => {
      let codeToCountryConverter= new countryConverter();
      expect(codeToCountryConverter.transform("IL")).toBe("Israel");
  
    });
});