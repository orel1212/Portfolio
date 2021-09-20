using Project.DAL;
using Project.ModelBinders;
using Project.Models;
using Project.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Security;

namespace Project.Controllers
{
    public class UserController : Controller
    {
        [HttpGet]
        public ActionResult Authenticate()
        {
            return View("Login");
        }
        [HttpPost]
        public ActionResult Authenticate([ModelBinder(typeof(UserBinder))]User user)
        {
            string msg = "";
            string role = "";
            if (user != null && ModelState.IsValid)
            {
                DataLayer dal = new DataLayer();
                List<Person> persons = (from p in dal.Persons where (p.ID.Equals(user.ID) && p.Password.Equals(user.Password)) select p).ToList<Person>();//LINQ
                if (persons.Count == 1)
                {
                    FormsAuthentication.SetAuthCookie(persons[0].ID,true);
                    if (persons[0].Role.Equals("Cashier"))
                        role="Cashier";
                    else if (persons[0].Role.Equals("Manager"))
                        role="Manager";
                    else
                        role="Customer";
                }
                else
                   msg= "Invalid user name or password!";
            }
            else
                msg = "Invalid user name or password!";
            var json = new { role, msg };
            return Json(json, JsonRequestBehavior.AllowGet);
        }
        [Authorize(Roles = "Manager,Cashier,Customer")]
        [HttpGet]
        public ActionResult EditInfo()
        {
            string id = getCookieID();
            Person person;
            DataLayer dal = new DataLayer();
            List<Person> per = (from x in dal.Persons where (x.ID.Equals(id)) select x).ToList<Person>();
            if (per[0].Role.Equals("Manager"))
            {
                person = new Manager(per[0].ID, per[0].Name, "");
            }
            else if (per[0].Role.Equals("Cashier"))
            {
                person = new Cashier(per[0].ID, per[0].Name, "");
            }
            else
                person = new Customer(per[0].ID, per[0].Name, "");
            return View("EditInfo", person);
        }
        [Authorize(Roles = "Manager,Cashier,Customer")]
        [HttpPost]
        public ActionResult EditInfo([ModelBinder(typeof(PersonBinder))]Person person)
        {

            string msg = "";
            DataLayer dal = new DataLayer();
            if (TryValidateModel((Person)person))
            {
                List<Person> p = (from x in dal.Persons where (x.ID.Equals(person.ID)) select x).ToList<Person>();
                p[0].Name = person.Name;
                p[0].Password = person.Password;
                dal.SaveChanges();
            }
            else
                msg = "All fields must be valid!";
            var json = new { msg, role = person.Role };
            return Json(json, JsonRequestBehavior.AllowGet);
        }
         [Authorize(Roles = "Manager,Cashier,Customer")]
         private string getCookieID()
         {
             string cookieName = FormsAuthentication.FormsCookieName; //Find cookie name
             HttpCookie authCookie = HttpContext.Request.Cookies[cookieName]; //Get the cookie by it's name
             FormsAuthenticationTicket ticket = FormsAuthentication.Decrypt(authCookie.Value); //Decrypt it
             return ticket.Name; //You have the UserName!
         }
    }
}