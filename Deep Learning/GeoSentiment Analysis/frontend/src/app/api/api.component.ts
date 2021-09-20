import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {
  domain:string="timorsatarov.me";
  protocol:string="https";
  port:number=443;
  constructor() { }

  ngOnInit() {
  }

}
