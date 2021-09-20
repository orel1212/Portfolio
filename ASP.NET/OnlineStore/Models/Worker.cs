using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace Project.Models
{

    public abstract class Worker:Person
    {
        public Worker(string id, string name, string password, string Role)
            : base(id, name, password, Role)
            {
            }

    }
}