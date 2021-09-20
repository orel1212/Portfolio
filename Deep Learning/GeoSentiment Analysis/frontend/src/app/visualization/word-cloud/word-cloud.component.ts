import { Component, OnInit } from '@angular/core';
import { CloudData, CloudOptions } from 'angular-tag-cloud-module';
import { tweetsService } from '../../shared/tweets.service';

@Component({
  selector: 'app-word-cloud',
  templateUrl: './word-cloud.component.html',
  styleUrls: ['./word-cloud.component.css']
})
export class WordCloudComponent implements OnInit {
  options: CloudOptions = {
    // if width is between 0 and 1 it will be set to the size of the upper element multiplied by the value 
    width : 1,
    height : 300,
    overflow: true,
  }
  noWordsFound:boolean = false;
  finishedCalculations:boolean=false;
  data: CloudData[];

  constructor(private tweetsService: tweetsService) { }

  ngOnInit() {
    this.data = [];
    setTimeout(()=>{this.createWordCloudData();},5);
  }

  createWordCloudData() {
    let words=this.tweetsService.getWords();
    let words_values = Object.keys(words).map(function(key) {
      return [key, words[key]];
    });
    if (words_values.length > 70)
    {
      if(words_values.length>1000)
      {
        let filter_value=3;
        if(words_values.length>3000)
        {
          if(words_values.length>10000)
          {
            if(words_values.length>20000)
            {
              if(words_values.length>40000)
              {
                filter_value=10;
              }
              else
              {
                filter_value=8;
              }

            }
            else
            {
              filter_value=6;
            }
          }
          else
          {
            filter_value=4;
          }
        }
        words_values = words_values.filter(function(words_values) {
          return words_values[1]>filter_value;
        });
      }
      words_values.sort((one, two) => (one[1] > two[1] ? -1 : 1));
      words_values=words_values.slice(0, 70);
    }
    for (let word_value of words_values) {
      let text:string = word_value[0].toString();
      let weight: number = parseInt(word_value[1].toString());
      this.data.push({text:text ,weight:Math.log(weight)});
    }
    if(words_values.length===0)
    {
      this.noWordsFound=true;
    }
    this.finishedCalculations=true;
  }

}
