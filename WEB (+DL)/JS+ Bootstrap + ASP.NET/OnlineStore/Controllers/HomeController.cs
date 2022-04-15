using Project.DAL;
using Project.ModelBinders;
using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Routing;
using System.Web.Security;

namespace Project.Controllers
{
    
    public class HomeController : Controller
    {
        public ActionResult Home()
        {
            return View(new User());
        }

        [HttpGet]
        public ActionResult Registration()
        {
            return View("Registration", new Customer());
        }
        [HttpPost]
        public ActionResult Registration([ModelBinder(typeof(CustomerBinder))] Customer cust)
        {
            DataLayer dal = new DataLayer();
            string msg = "";
            if (cust != null && TryValidateModel((Person)cust))
            {
                List<Person> persons = (from p in dal.Persons where p.ID.Equals(cust.ID) select p).ToList<Person>();//LINQ
                if (persons.Count == 0)
                {
                    dal.Persons.Add(cust);//in memory ,without commit to the db
                    dal.SaveChanges();

                }
                else
                    msg = "the id must be unique,the id already exist in the system!";
            }
            else
                msg = "Wrong input!";

            return Json(msg, JsonRequestBehavior.AllowGet);
        }
        
        [Authorize(Roles="Manager,Cashier,Customer")]
        [HttpPost]
        public ActionResult Logout()
        {
            FormsAuthentication.SignOut();
            return RedirectToAction("Home");
        }
    }
}