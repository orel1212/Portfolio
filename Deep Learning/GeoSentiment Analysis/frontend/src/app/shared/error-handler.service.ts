export class errorHandlerService
{
    private message:string;
    private code:number;
    isError:boolean;
    constructor()
    {
        this.clearError();
    }
    getMessage()
    {
        return this.message;
    }
    getCode()
    {
        return this.code;
    }
    setMessage(msg:string)
    {
        this.isError=true;
        this.message=msg;
    }
    setCode(c:number)
    {
        this.isError=true;
        this.code=c;
    }
    clearError()
    {
        this.message="";
        this.code=0;
        this.isError=false;
    }


}