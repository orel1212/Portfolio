export class Country {
    positive = 0;
    negative = 0;
    languages: {[language: string] : number} = {};
    constructor(public name){
    }

    addLanguage(lang: string){
        if (this.languages[lang] == undefined) {
            this.languages[lang] = 1;
        }
        else {
            this.languages[lang] +=1;
        }
    }

    incPositive() {
        this.positive += 1;
    }

    incNegative() {
        this.negative += 1;
    }

    getPositive() {
        return this.positive;
    }

    getNegative() {
        return this.negative;
    }

    getLanguages() {
        return this.languages;
    }
}