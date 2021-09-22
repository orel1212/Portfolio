import { Component, OnInit } from '@angular/core';
import { tweetsService } from '../../shared/tweets.service';
import { Router } from '@angular/router';
import {saveAs} from "file-saver";
import { countryConverter } from '../../shared/countryConverter.pipe';

@Component({
  selector: 'app-statistics',
  templateUrl: './statistics.component.html',
  styleUrls: ['./statistics.component.css']
})
export class StatisticsComponent implements OnInit {
  countryStats: {
                          countryCode:string,
                          countryName:string,
                          totalTweets:number,
                          positiveTweets:number,
                          negativeTweets:number,
                          avgSentiment:string
                        }[] =[];
  private countryConvert: countryConverter= new countryConverter();

  constructor(
              private tweetsService: tweetsService,
              private router: Router){}

  ngOnInit() {
    let countries = this.tweetsService.getCountries();
    for (let country in countries){
      let countryCode = country;
      let countryName = this.countryConvert.transform(country);
      let positiveTweets = countries[country].getPositive();
      let negativeTweets = countries[country].getNegative();
      let totalTweets = positiveTweets + negativeTweets;
      let positivePercentage = positiveTweets/totalTweets;
      let avgSentiment = "positive";
      if (positivePercentage < 0.5){
        avgSentiment = "negative";
      }
      else if ( positivePercentage === 0.5 ){
        avgSentiment = "neutral";
      }
      this.countryStats.push({countryCode,countryName, totalTweets, positiveTweets, negativeTweets, avgSentiment});
    }
  }


  onClickCountry(code: string){
    this.router.navigate(['/visualization/lang-cloud'], {queryParams : {'country' : code}});
  }

  downloadStatsAsJson() {
    let filename = 'statistics.json'
    let blob = new Blob([JSON.stringify(this.countryStats, null, 2)], { type: 'application/json' });
    saveAs(blob, filename);
  }


}
