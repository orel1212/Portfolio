import { Component, OnInit } from '@angular/core';
import { CloudData, CloudOptions } from 'angular-tag-cloud-module';
import { tweetsService } from '../../shared/tweets.service';

@Component({
  selector: 'app-hashtag-cloud',
  templateUrl: './hashtag-cloud.component.html',
  styleUrls: ['./hashtag-cloud.component.css']
})
export class HashtagCloudComponent implements OnInit {
  options: CloudOptions = {
    // if width is between 0 and 1 it will be set to the size of the upper element multiplied by the value 
    width : 1,
    height : 300,
    overflow: true,
  }
  finishedCalculations:boolean=false;
  noHashtagsFound:boolean = false;
  data: CloudData[];
  
  constructor(private tweetsService: tweetsService) { }

  ngOnInit() {
    this.data = [];
    setTimeout(()=>{this.createHashtagCloudData();},5);
  }

  createHashtagCloudData() {
    let hashtags=this.tweetsService.getHashtags();
    let keys=Object.keys(hashtags);
    let hashtags_values = keys.map(function(key) {
      return [key, hashtags[key]];
    });
    if (hashtags_values.length > 30)
    {
      
      if(hashtags_values.length>250)
      {
        let filter_value=1;
        if(hashtags_values.length>1000)
        {
          if(hashtags_values.length>2000)
          {
            if(hashtags_values.length>4000)
            {
              if(hashtags_values.length>5000)
              {
                filter_value=5;
              }
              else
              {
                filter_value=4;
              }

            }
            else
            {
              filter_value=3;
            }
          }
          else
          {
            filter_value=2;
          }
        }

        hashtags_values = hashtags_values.filter(function(hashtag_value) {
          return hashtag_value[1]>filter_value;
        });
      }
      
      hashtags_values.sort((one, two) => (one[1] > two[1] ? -1 : 1));
      hashtags_values=hashtags_values.slice(0, 30);
    }
    for (let hashtag_value of hashtags_values) {
      let text:string = hashtag_value[0].toString();
      let weight: number = parseInt(hashtag_value[1].toString());
      this.data.push({text:text ,weight:Math.log(weight)});
    }
    if(keys.length===0)
    {
      this.noHashtagsFound=true;
    }
    this.finishedCalculations=true;
  }

}
