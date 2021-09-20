import { Component, OnInit } from '@angular/core';
import { CloudData, CloudOptions } from 'angular-tag-cloud-module';
import { tweetsService } from '../../shared/tweets.service';
import { Country } from '../../shared/country.model';
import { ActivatedRoute, Router } from '@angular/router';
import { languageConverter } from '../../shared/langConverter.pipe';

@Component({
  selector: 'app-lang-cloud',
  templateUrl: './lang-cloud.component.html',
  styleUrls: ['./lang-cloud.component.css']
})
export class LangCloudComponent implements OnInit {


  private countries: {[name: string] : Country } = {};
  private totalLanguagesCount: {[name: string] : number} = {};
  public country: string;
  options: CloudOptions = {
    // if width is between 0 and 1 it will be set to the size of the upper element multiplied by the value 
    width : 1,
    height : 300,
    overflow: true,
  }
  langConverter: languageConverter= new languageConverter();

  data: CloudData[];

  constructor(
              private route: ActivatedRoute,
              private tweetsService: tweetsService,
              private router: Router) { }

  ngOnInit() {
    this.totalLanguagesCount = { };
    this.data = [];
    this.countries = this.tweetsService.getCountries();
    this.country = this.route.snapshot.queryParams['country'];
    this.country = this.country.toLowerCase();
    this.route.queryParams
      .subscribe(
        (params) => {
          this.country = params['country'];
          this.country = this.country.toLowerCase();
          this.initWordCloud();
        }
      );
  }

  initWordCloud(){
    this.data = [];
    if (this.country === 'all') {
      this.createGlobalWordCloud();
    }
    else {
      if (this.countries[this.country] == undefined){
        this.router.navigate(['/not-found']);   
      }
      else {
        this.createCountryWordCloud(this.country);
      }
    }
  }

  addLanguage(lang: string, count: number){
    if (this.totalLanguagesCount[lang] == undefined) {
        this.totalLanguagesCount[lang] = count;
    }
    else {
        this.totalLanguagesCount[lang] += count;
    }
}

  countCountryLanguages(name: string) {
    let country = this.countries[name];
    let languages = country.getLanguages();
    for (let lang in languages) {
      this.addLanguage(lang,languages[lang]);
    }
  }

  createWordCloudData() {
    for (let lang in this.totalLanguagesCount) { 
      let entry = {text: this.langConverter.transform(lang), weight: Math.log(this.totalLanguagesCount[lang])};
      this.data.push(entry);
    }
  }

  createCountryWordCloud(name: string){
    this.totalLanguagesCount = { };
    this.countCountryLanguages(name);
    this.createWordCloudData();
  }

  createGlobalWordCloud() {
    this.totalLanguagesCount = { };
    for (let country in this.countries) {
      this.countCountryLanguages(country);
    }
    this.createWordCloudData();
  }
  getCountries()
  {
    return this.countries;
  }

}
