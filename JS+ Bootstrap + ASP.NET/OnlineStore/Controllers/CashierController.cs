using Project.DAL;
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
    [Authorize(Roles="Cashier")]
    public class CashierController : Controller
    {
        public ActionResult CashierHome()
        {
            return View();
        }
        public ActionResult GetOpennedOrdersByJson()
        {
            DataLayer dal = new DataLayer();
            List<Order> opennedOrders = (from x in dal.Orders where (x.Status.Equals("Openned")) select x).ToList<Order>();
            return Json(opennedOrders, JsonRequestBehavior.AllowGet);
        }
        [HttpGet]
        public ActionResult CompleteOrder()
        {
            return View("CompleteOrder");
        }
        [HttpPost]
        public ActionResult CompleteOrder(int oid)
        {
            DataLayer dal = new DataLayer();
            Complete(oid,dal);
            return RedirectToAction("CashierHome");
        }
        public ActionResult CompleteAllOrders()
        {
            DataLayer dal = new DataLayer();
            List<Order> opennedOrders = (from x in dal.Orders where (x.Status.Equals("Openned")) select x).ToList<Order>();
            foreach(Order od in opennedOrders)
                Complete(od.OID,dal);
            return RedirectToAction("CashierHome");
        }
        private void Complete(int oid,DataLayer dal)
        {
            List<Order> opennedOrder = (from x in dal.Orders where (x.OID == oid) select x).ToList<Order>();
            List<OrderDescription> ODs = (from x in dal.ODs where (x.OID == oid) select x).ToList<OrderDescription>();
            foreach (OrderDescription od in ODs)
            {
                List<Product> product = (from x in dal.Products where (x.PID == od.PID) select x).ToList<Product>();
                if (product[0].Amount > od.Amount)
                {
                    product[0].Amount -= od.Amount;
                }
                else
                {
                    product[0].Amount = product[0].Amount + od.Amount;
                    product[0].Amount -= od.Amount;
                }
            }
            opennedOrder[0].Status = "Completed";
            dal.SaveChanges();
        }

        /*
        [HttpGet]
        public ActionResult EditInfo()
        {
            string id = getCookieID();
            return View("Person/EditInfo", new PersonVM(id));
        }
        [HttpPost]
        public ActionResult EditInfo(Cashier cash)
        {
            string msg = "";
            DataLayer dal = new DataLayer();
            if (TryValidateModel((Person)cash))
            {
                List<Person> cashier = (from x in dal.Persons where (x.ID.Equals(cash.ID)) select x).ToList<Person>();
                cashier[0].Name = cash.Name;
                cashier[0].Password = cash.Password;
                dal.SaveChanges();
            }
            else
                msg = "All fields must be valid!";
            var json = new { msg, role = "Cashier" };
            return Json(json, JsonRequestBehavior.AllowGet);
        }
         */
        private string getCookieID()
        {
            string cookieName = FormsAuthentication.FormsCookieName; //Find cookie name
            HttpCookie authCookie = HttpContext.Request.Cookies[cookieName]; //Get the cookie by it's name
            FormsAuthenticationTicket ticket = FormsAuthentication.Decrypt(authCookie.Value); //Decrypt it
            return ticket.Name; //You have the UserName!
        }
    }
}