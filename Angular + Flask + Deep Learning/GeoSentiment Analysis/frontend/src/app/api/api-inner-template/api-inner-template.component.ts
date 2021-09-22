import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-api-inner-template',
  templateUrl: './api-inner-template.component.html',
  styleUrls: ['./api-inner-template.component.css']
})
export class ApiInnerTemplateComponent implements OnInit {
  title:String;
  url:String;
  method:String;
  urlParams=[];
  dataParams=[];
  headerParams=[];
  headers=[];
  successResponse={};
  errorResponses=[];
  @Input() docType:string;
  constructor() { }
  ngOnInit() {
    if(this.docType==='refreshToken')
    {
      this.headerParams.push({
        'name':"refresh_token",
        'type':"[JWT-token]"
        }
      );
      this.headers.push(
        {
          'name':"Authorization",
          'value':"Bearer refresh_token"
        }
      );
      this.title="Refresh JWT Token"
      this.url="/api/refresh_token";
      this.successResponse={
        "name":"access_token",
        "content":"access_token_jwt",
        "listItemFormat":"<code>access_token:[JWT-token]</code>"
      };
      this.method='GET';
      this.errorResponses.push({
        "errorCode":1,
        "errorMsg":"Internal Server Error"
        });
      this.errorResponses.push(
        {
        "errorCode":4,
        "errorMsg":"The Token Has Expired"
        });
      this.errorResponses.push({
      "errorCode":5,
      "errorMsg":"The Token is Invalid"
      });
      this.errorResponses.push({
        "errorCode":6,
        "errorMsg":"Unauthorized Access"
        });

    }
    else if(this.docType==='VerifyCode')
    {
      this.dataParams.push({
        'name':"access_token",
        'type':"[JWT-token]"
        }
      );
      this.title="POST Verification Code"
      this.url="/api/twitter_verify_auth";
      this.successResponse={
      };
      this.method='POST';
      this.headers.push(
        {
          'name':"Content-Type",
          'value':"application/json"
        }
      );
      this.errorResponses.push({
        "errorCode":1,
        "errorMsg":"Internal Server Error"
        });
      this.errorResponses.push(
        {
        "errorCode":10,
        "errorMsg":"Reverse auth credentials error"
        });
      this.errorResponses.push(
        {
        "errorCode":11,
        "errorMsg":"JSON Format Invalid"
        });
      this.errorResponses.push({
      "errorCode":12,
      "errorMsg":"No JSON given"
      });
    }
    else if(this.docType==='Auth')
    {
      this.title="Authentication"
      this.url="/api/twitter_auth";
      this.successResponse={
      };
      this.method='GET';
      this.errorResponses.push({
        "errorCode":1,
        "errorMsg":"Internal Server Error"
        });
    }
    else if (this.docType==='Trends' || this.docType==='demoTrends')
    {
      this.errorResponses.push({
        "errorCode":1,
        "errorMsg":"Internal Server Error"
        });

      
      if(this.docType==='Trends')
      {
        this.errorResponses.push(
          {
          "errorCode":4,
          "errorMsg":"The Token Has Expired"
          });
        this.errorResponses.push({
        "errorCode":5,
        "errorMsg":"The Token is Invalid"
        });
        this.errorResponses.push({
          "errorCode":6,
          "errorMsg":"Unauthorized Access"
          });
        this.errorResponses.push({
          "errorCode":8,
          "errorMsg":"Search Rate Limit"
          });
          this.errorResponses.push({
            "errorCode":9,
            "errorMsg":"Invalid or expired twitter access tokens"
          });
        this.title="Get Trending Hashtags - Online Mode"
        this.url="/api/trends";
        this.headerParams.push({
          'name':"access_token",
          'type':"[JWT-token]"
          }
        );
        this.headers.push(
          {
            'name':"Authorization",
            'value':"Bearer access_token"
          }
        );
      }
      else
      {
        this.title="Get Trending Hashtags - Demo Mode"
        this.url="/api/demo/trends";
      }
      this.method='GET';
      this.successResponse={
        "name":"trends",
        "content":"[list of hashtags]",
        "listItemFormat":"<code>#hashtagName</code>"
      };
      
    }
    else if(this.docType==='Predictions' || this.docType==='demoPredictions')
    {
      this.errorResponses.push({
        "errorCode":1,
        "errorMsg":"Internal Server Error"
        });
      if(this.docType==='Predictions')
      {
        this.errorResponses.push({
          "errorCode":2,
          "errorMsg":"Forbidden Input"
          });
        this.errorResponses.push(
          {
          "errorCode":4,
          "errorMsg":"The Token Has Expired"
          });
        this.errorResponses.push({
        "errorCode":5,
        "errorMsg":"The Token is Invalid"
        });
        this.errorResponses.push({
          "errorCode":6,
          "errorMsg":"Unauthorized Access"
          });
        this.errorResponses.push({
          "errorCode":8,
          "errorMsg":"Search Rate Limit"
          });
          this.errorResponses.push({
            "errorCode":9,
            "errorMsg":"Invalid or expired twitter access tokens"
          });
        this.title="Get Predictions - Online Mode"
        this.url="/api/predictions/:input";
        this.headerParams.push({
          'name':"access_token",
          'type':"[JWT-token]"
          }
        );
        this.headers.push(
          {
            'name':"Authorization",
            'value':"Bearer access_token"
          }
        );
      }
      else
      {

        this.title="Get Predictions - Demo Mode"
        this.url="/api/demo/predictions/:input";
        this.errorResponses.push({
          "errorCode":2,
          "errorMsg":"Forbidden Input"
          });
      }
      
      this.method='GET';
      this.urlParams.push(
        {
          'name':"input",
          'type':"[string]"
        }
      );
      this.successResponse={
        "name":"tweets",
        "content":"[list of tweets]",
        "listItemFormat":"<code>country_prediction : cc(country code)<br>text : tweet content<br>created_at : Date<br>sentiment : positive/negative<br>tweet_lang : tweet_language</code>"
        };
    }
  }

}
