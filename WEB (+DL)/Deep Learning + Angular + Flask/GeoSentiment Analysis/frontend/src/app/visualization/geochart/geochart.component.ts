import { Component, OnInit } from '@angular/core';
import { tweetsService } from '../../shared/tweets.service';
import { countryConverter } from '../../shared/countryConverter.pipe';

@Component({
  selector: 'app-geochart',
  templateUrl: './geochart.component.html',
  styleUrls: ['./geochart.component.css']
})
export class GeochartComponent implements OnInit {
  mapChartData;
  private countryConvert: countryConverter= new countryConverter();
  constructor(private tweetsService: tweetsService) { }

  ngOnInit() {
    this.mapChartData = this.createMap();
  }
  
  createMap() {
    
    let countries = this.tweetsService.getCountries();
    let chartType = 'GeoChart';
    let dataTable = [];
    let tmp = ['Country', 'Positive %', {'type': 'string', 'role': 'tooltip', 'p': {'html': true}}];
    dataTable.push(tmp);
    for (let country in countries){
      let positiveTweets = countries[country]['positive'];
      let negativeTweets = countries[country]['negative'];
      let totalTweets = positiveTweets + negativeTweets;

      let positivePercentage = positiveTweets/totalTweets*100;
      
      let tmp = [{v:country,f:this.countryConvert.transform(country)}, positivePercentage,this.createCustomHTMLContent(positiveTweets,negativeTweets) ];
      dataTable.push(tmp);
    }
    let options =  {
      colorAxis: {colors: ['red', 'green'], minValue: 0, maxValue: 100 },
      tooltip: {isHtml: true},
      focusTarget: 'category'
    };

    let mapChartData =  {
      chartType: chartType,
      dataTable: dataTable,
      options: options
    };
    return mapChartData;
  }

  createCustomHTMLContent( positive:number, negative:number) {
    let numberOfTweets:string = '<p>Total tweets: <strong>' + (positive+negative) + '</strong></p>';
    let numberOfPositive:string = '<p>Positive tweets: <strong>' + positive + '</strong></p>';
    let numberOfNegative:string = '<p>Negative tweets: <strong>' + negative + '</strong></p>';
    let contentHTML:string = '<div>' + numberOfTweets + numberOfPositive + numberOfNegative + '</div>';
    return contentHTML;
  }

}
