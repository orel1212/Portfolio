import { Component, OnInit } from '@angular/core';
import { tweetsService } from '../../shared/tweets.service';
import { Tweet } from '../../shared/tweet.model';

@Component({
  selector: 'app-linechart',
  templateUrl: './linechart.component.html',
  styleUrls: ['./linechart.component.css']
})
export class LinechartComponent implements OnInit {

  lineChartData;
  constructor(private tweetsService: tweetsService) { }

  ngOnInit() {
    this.lineChartData = this.createLineChart();
  }
  getShapedDate(date:Date)
  {
      let day=date.getDate();
      let month=date.getMonth()+1;
      let year=date.getFullYear();
      return ''+day+'/'+month+'/'+year;
  }
  getSortedDates()
  {
    let tweets:Tweet[] = this.tweetsService.getTweets();
    
    let dict_dates={};
    for (let tweet of tweets)
    {
      let date=tweet.createdAt;
      let day=date.getDate();
      let month=date.getMonth()+1;
      let year=date.getFullYear();
      let date_key=''+day+'/'+month+'/'+year;
      if (dict_dates[date_key]===undefined)
      {
        dict_dates[date_key]={
          'dateObj':new Date(year, month-1, day, 0, 0, 0, 0),
          'numOfTweets':1
        };
      }
      else
      {
        dict_dates[date_key]['numOfTweets']+=1;
      }
    }
    let datesArray=Object.values(dict_dates);
    datesArray.sort((a, b) => new Date(a['dateObj']).getTime() - new Date(b['dateObj']).getTime());
    return datesArray;
  }
  createLineChart() {
    
    let chartType = 'LineChart';
    let dataTable = [];
    let tmp = [{label: 'Date', type: 'string'},
               {label: 'Total tweets', type: 'number'}
              ]
    dataTable.push(tmp);
    let datesArray=this.getSortedDates();
    for(let date of datesArray)
    {
      let shaped_date=this.getShapedDate(date['dateObj']);
      let tweets_num:number=date['numOfTweets'];
      let tmp = [shaped_date, tweets_num];
      dataTable.push(tmp);
    }
    let options =  {
      tooltip: {isHtml: true},
      curveType: 'function',
      legend: { position: 'bottom' },
      vAxis: { format: '0'}
    };

    let ChartData =  {
      chartType: chartType,
      dataTable: dataTable,
      options: options
    };
    return ChartData;
  }

}
