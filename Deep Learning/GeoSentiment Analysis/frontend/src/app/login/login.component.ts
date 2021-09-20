import { Component, OnInit } from '@angular/core';
import { apiService } from '../shared/api.service';
import { FormGroup, Validators, FormControl } from '@angular/forms';
import { errorHandlerService } from '../shared/error-handler.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  jsonTokens;
  url=undefined;
  isLoadingURL:boolean=true;
  errorMsg:string="";
  isCodeError:boolean = false;
  verifierCodeForm: FormGroup;
  constructor(private apiService: apiService, private errorHandler:errorHandlerService, private router: Router) { }

  ngOnInit() {
    this.initForm();
    this.apiService.requestTwitterAuthUrl()
    .subscribe(
      (response) => {
        
        this.jsonTokens = response;
        this.url = this.jsonTokens['request_url'];
        this.isLoadingURL=false;
        this.errorMsg="";
      },
      (error) => 
      {
        this.isLoadingURL=false;
        let errorCode=this.errorHandler.getCode();
        if(errorCode===1)
        {
          this.errorMsg = "Login functionallity is not working at the moment.";
        }
        else
        {
          this.errorMsg = this.errorHandler.getMessage();
        }
       }
    );
  }

  private initForm() {
    let codeInput = '';
    let codeFormObject={};
    let validatorsArray=[Validators.required];
    validatorsArray.push(Validators.pattern("[0-9]+$"));
    codeFormObject['codeInput']= new FormControl(codeInput, validatorsArray);
    this.verifierCodeForm = new FormGroup(codeFormObject);
  }
  onSubmit()
  {
    if (this.verifierCodeForm.valid)
    {
      let codeInput = this.verifierCodeForm.get("codeInput").value;
      
      this.jsonTokens['verifier_code'] = codeInput;
      this.apiService.postTwitterVerifierCode(this.jsonTokens)
      .subscribe(
        (response) => {
          this.isCodeError = false;
          this.router.navigate(['/search'], { queryParams: { mode: 'online' } });
        },
        (error) => { 
          this.errorMsg = this.errorHandler.getMessage();
          this.isCodeError = true;}
      );

    }
  
  }
}
