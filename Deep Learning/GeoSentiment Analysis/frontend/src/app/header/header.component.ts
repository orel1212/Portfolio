import { Component, OnInit } from '@angular/core';
import { apiService } from '../shared/api.service';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {

  collapse:boolean = true;
  
  constructor() { }

  ngOnInit() {
  }

  toggleCollapse() {
    this.collapse = !this.collapse;
  }
  

}
