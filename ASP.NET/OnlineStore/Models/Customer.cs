using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class Customer:Person
    {
        public Customer(string id, string name,string password)
            : base(id, name,password,"Customer")
        {
            
        }

        public Customer()
            : base("", "", "", "Customer")
        {
            
        }
        public Customer(Person p)
            : base(p.ID, p.Name, p.Password, "Customer")
        {

        }
    }
}