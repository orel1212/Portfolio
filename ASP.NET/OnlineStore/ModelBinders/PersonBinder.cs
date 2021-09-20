using Project.DAL;
using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Project.ModelBinders
{
    public class PersonBinder : IModelBinder
    {
            public object BindModel(ControllerContext controllerContext, ModelBindingContext bindingContext)
            {
                HttpContextBase objContext = controllerContext.HttpContext;
                string id = objContext.Request.Form["id"].ToString();
                string name = objContext.Request.Form["name"].ToString();
                string pass = objContext.Request.Form["password"].ToString();
                Person person;
                DataLayer dal = new DataLayer();
                List<Person> per = (from x in dal.Persons where (x.ID.Equals(id)) select x).ToList<Person>();
                if (per[0].Role.Equals("Manager"))
                {
                    person = new Manager(id,name,pass);
                }
                else if (per[0].Role.Equals("Cashier"))
                {
                    person = new Cashier(id, name, pass);
                }
                else
                    person = new Customer(id, name, pass);
                return person;
            }
    }
}