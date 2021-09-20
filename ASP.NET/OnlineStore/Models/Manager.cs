using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Project.Models
{

    public class Manager:Worker
    {
        public Manager(string id, string name, string password)
            : base(id, name, password, "Manager")
            {
            }
        public Manager()
            : base("", "", "", "Manager")
        {
            
        }
        public Manager(Person p)
            : base(p.ID, p.Name, p.Password, "Manager")
        {

        }
    }
}