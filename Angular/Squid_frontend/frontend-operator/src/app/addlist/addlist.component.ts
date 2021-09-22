import { Component, OnInit, OnDestroy } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { apiService } from '../api.service';

@Component({
  selector: 'app-addlist',
  templateUrl: './addlist.component.html',
  styleUrls: ['./addlist.component.css']
})
export class AddlistComponent implements OnInit,OnDestroy {
  chosensList: string[] = [];
  selectedIndex: number = -1;
  listForm: FormGroup;
  preScrollable: boolean = true;



  constructor(
    private router: Router,
    private apiService: apiService
  ) { }
  ngOnInit(): void {
    this.chosensList = [];
    this.initForm();
    this.selectedIndex = -1;
    this.preScrollable = true;
  }
  onClick(i:number){
    this.selectedIndex = i;
  }

  isSelected(i:number){
    return this.selectedIndex===i;
  }

  formSubmit() {
    if (this.listForm.valid){
      let listInput = this.listForm.get("listInput").value;
      if(this.chosensList.indexOf(listInput) === -1)
      {
        this.chosensList.push(listInput)
      }
      this.initForm();
    }
  }

  onSubmit()
  {
    this.apiService.setList(this.chosensList);
    this.chosensList = [];
    this.selectedIndex = -1;
    this.router.navigate(['/status']);
  }
  removeChosen() 
  {
    if (this.selectedIndex > -1) {
      this.chosensList.splice(this.selectedIndex, 1);
      this.selectedIndex = -1;
    }
  }


  private validateInput(c: FormControl) {
    let URL_PATTERN = /^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$/; // Regular Expression 1
    let IP_PATTERN = /^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$/;// Regular Expression 2
    return (URL_PATTERN.test(c.value) || IP_PATTERN.test(c.value)) ? null : {
      validateInput: {
        valid: false
      }
    };
  }
  private initForm() {
    let listInput = '';
    let listFormObject={};
    let validatorsArray=[Validators.required];
    validatorsArray.push(this.validateInput);
    listFormObject['listInput']= new FormControl(listInput, validatorsArray);
    this.listForm = new FormGroup(listFormObject);
  }

  ngOnDestroy() : void
  {
  }


}
