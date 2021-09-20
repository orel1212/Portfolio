using Project.DAL;
using Project.ModelBinders;
using Project.Models;
using Project.ViewModels;
using System;
using System.Collections.Generic;
using System.Data.Entity.Validation;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Security;

namespace Project.Controllers
{
    
    public class ManagerController : Controller
    {
        public ActionResult GetProductsByJson(string Msg)
        {
            DataLayer dal = new DataLayer();
            List<Product> Products = dal.Products.ToList<Product>();
            var json = new { msg = Msg, Products };
            return Json(json, JsonRequestBehavior.AllowGet);
        }
        [Authorize(Roles = "Manager")]
        public ActionResult ManagerHome()
        {
            return View();
        }
        [Authorize(Roles = "Manager")]
        [HttpGet]
        public ActionResult AddWorker()
        {
            return View("AddWorker");
        }
        [Authorize(Roles = "Manager")]
        [HttpPost]
        public ActionResult AddWorker([ModelBinder(typeof(WorkerBinder))] Worker worker)
        {
            string msg = "";
            DataLayer dal = new DataLayer();
            if (worker != null && TryValidateModel((Person)worker))
            {
               
                List<Person> persons = (from p in dal.Persons where p.ID.Equals(worker.ID) select p).ToList<Person>();//LINQ
                if (persons.Count==0)
                {
                    dal.Persons.Add(worker);//in memory ,without commit to the db
                    dal.SaveChanges();
                }
                else
                {
                    msg= "the id must be unique,the id already exist in the system!";
                }
                    
            }
            else
                msg = "Wrong input!";
            return GetWorkersByJson(msg);
        }
        [Authorize(Roles = "Manager")]
         [HttpGet]
        public ActionResult AddProduct()
        {
            return View("AddProduct", new Product());
        }
        [Authorize(Roles = "Manager")]
        [HttpPost]
        public ActionResult AddProduct([ModelBinder(typeof(ProductBinder))] Product product)
        {
            string msg = "";
            DataLayer dal = new DataLayer();
            bool check=TryValidateModel(product);
            if (product != null && check)
            {
                    dal.Products.Add(product);//in memory ,without commit to the db
                    dal.SaveChanges();
            }
            else
                msg = "Invalid Input!";
            return GetProductsByJson(msg);
        }
        [Authorize(Roles = "Manager")]
        public ActionResult GetWorkersByJson(string Msg)
        {
            string id = getCookieID();
            string msg = Msg;
            DataLayer dal = new DataLayer();
            //returning all the workers besides him!(he knows his id and to prevent from deleting himself)
            List<Person> workers = (from x in dal.Persons where (!x.ID.Equals(id) && (x.Role.Equals("Cashier") || x.Role.Equals("Manager"))) select x).ToList<Person>();
            var json = new { Workers = workers, msg };
            return Json(json, JsonRequestBehavior.AllowGet);
        }
        
        [Authorize(Roles = "Manager")]
        [HttpGet]
        public ActionResult ViewProfits()
        {
            return View();
        }
        [Authorize(Roles = "Manager")]
        [HttpGet]
        public ActionResult GetCompletedOrdersByJson()
        {
            DataLayer dal = new DataLayer();
            List<Order> completedOrders = (from x in dal.Orders where (x.Status.Equals("Completed")) select x).ToList<Order>();
            return Json(completedOrders, JsonRequestBehavior.AllowGet);
        }
        [Authorize(Roles = "Manager")]
        [HttpGet]
        public ActionResult DeleteWorker()
        {
            return View("DeleteWorker");
        }
        [Authorize(Roles = "Manager")]
        [HttpPost]
        public ActionResult DeleteWorker(string id)
        {
            DataLayer dal = new DataLayer();
            dal.Persons.RemoveRange(dal.Persons.Where(x => x.ID.Equals(id)));
            dal.SaveChanges();
            return RedirectToAction("ManagerHome");
        }
        [Authorize(Roles = "Manager")]
        [HttpGet]
        public ActionResult DeleteProduct()
        {
            return View("DeleteProduct");
        }
        [Authorize(Roles = "Manager")]
        [HttpPost]
        public ActionResult DeleteProduct(int pid)
        {
            DataLayer dal = new DataLayer();
            dal.ODs.RemoveRange(dal.ODs.Where(x => x.PID == pid));//foreign key constrait,it will affect the total of the order??
            dal.Products.RemoveRange(dal.Products.Where(x => x.PID==pid));
            dal.SaveChanges();
            return RedirectToAction("ManagerHome");
        }
        private string getCookieID()
        {
            string cookieName = FormsAuthentication.FormsCookieName; //Find cookie name
            HttpCookie authCookie = HttpContext.Request.Cookies[cookieName]; //Get the cookie by it's name
            FormsAuthenticationTicket ticket = FormsAuthentication.Decrypt(authCookie.Value); //Decrypt it
            return ticket.Name; //You have the UserName!
        }
        
    }
}