using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class Cashier:Worker
    {
        public Cashier(string id, string name, string password)
                : base(id, name, password, "Cashier")
            {
            }
        public Cashier()
            : base("", "", "", "Cashier")
        {
            
        }
        public Cashier(Person p)
            : base(p.ID, p.Name, p.Password, "Cashier")
        {

        }
    }
}