import numpy as np
from operator import *
from copy import deepcopy

class OperatorList:
    def __init__(self,size):
        self.size = size
        self.factor = 1

    def append(self,i,operator):
        exec('self.op{} = operator'.format(i))

    def get(self,i):
        return(eval('self.op{}'.format(i)))

    def replace(self,i,operator):
        A = deepcopy(self)
        exec('A.op{} = operator'.format(i))
        return(A)

    def __mul__(self,obj2):
        new_list = OperatorList(self.size)
        for i in range(self.size):
            op = self.get(i)*obj2.get(i)
            if op == 0:
                return(0)
            else:
                new_list.append(i,op)
        return(new_list)
            
    def __eq__(self,obj2):
        for i in range(self.size):
            if self.get(i).op != obj2.get(i).op:
                return(False)
        return(True)

    def calculate_factor(self):
        f = complex(1,0)
        for i in range(self.size):
            if self.get(i).im:
                f *= complex(0,self.get(i).factor)
            else:
                f *= complex(self.get(i).factor,0)
        self.factor = f

        

    def defactor(self):
        check = True
        tot = [self]
        while check:
            counter = 0
            tot2 = []
            for oplist in tot:
                for i in range(self.size):
                    if type(oplist.get(i)) == list:
                        list1 = oplist.replace(i,self.get(i)[0])
                        list2 = oplist.replace(i,self.get(i)[1])
                        tot2.append(list1)
                        tot2.append(list2)
                        counter += 1
                        break
            
            if counter == 0:
                check = False
            else:
                tot = tot2
        return(tot)
            

class Hamiltonian:
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits

    def remove_identity(self,res):
        n_qubits = self.n_qubits
        res2 = []
        for oplist in res:
            count = 0
            for i in range(n_qubits):
                if oplist.get(i).op == '':
                    count += 1
            if count != n_qubits:
                res2.append(oplist)
        return(res2)
    def calculate_factor(self,res):
        res2 = []
        for oplist in res:
            oplist.calculate_factor()
            if oplist.factor != 0:
                res2.append(oplist)
        return(res2)

    def add_equals(self,res):
        tot = []
        while len(res) > 0:
            res1 = deepcopy(res[0])
            f = res1.factor
            del res[0]
            res_copy = deepcopy(res)
            del_list = []
            for j in range(len(res_copy)):
                res2 = res_copy[j]
                if res1 == res2:
                    f += res2.factor

                    del_list.append(j)
            for i in sorted(del_list, reverse=True):
                del res[i]
            res1.factor = f
            if f != 0:
                tot.append(res1)
        return(tot)



    def get_circuits(self,h_pq,h_pqrs,remove_identity=False):
        n_qubits = self.n_qubits
        result = []
        for p in range(n_qubits):
            for q in range(n_qubits):
                if h_pq[p,q] != 0:

                    t_ob1 = OperatorList(n_qubits)
                    t_ob2 = OperatorList(n_qubits)

                    for i in range(p):
                        t_ob1.append(i,Operator('z'))
                    t_ob1.append(p,Operator('+'))
                    for i in range(p+1,self.n_qubits):
                        t_ob1.append(i,Operator('I'))

                    for i in range(q):
                        t_ob2.append(i,Operator('z'))
                    t_ob2.append(q,Operator('-'))
                    for i in range(q+1,self.n_qubits):
                        t_ob2.append(i,Operator('I'))
                    
                    res = t_ob1*t_ob2

                    
                    res = res.defactor()

                    if remove_identity:
                        res = self.remove_identity(res)

                    for oplist in res:
                        oplist.calculate_factor()

                    
                    res2 = []
                    for oplist in res:
                        oplist.factor *= h_pq[p,q]
                        res2.append(oplist)
                    
                    

                    result.extend(res2)

                for r in range(n_qubits):
                    for s in range(r+1,n_qubits):
                        if q <= p or h_pqrs[p,q,r,s] == 0:
                            continue

                        
                        t1 = OperatorList(n_qubits)
                        t2 = OperatorList(n_qubits)
                        t3 = OperatorList(n_qubits)
                        t4 = OperatorList(n_qubits)

                        for i in range(p):
                            t1.append(i,Operator('z'))
                        t1.append(p,Operator('+'))
                        for i in range(p+1,self.n_qubits):
                            t1.append(i,Operator('I'))

                        for i in range(q):
                            t2.append(i,Operator('z'))
                        t2.append(q,Operator('+'))
                        for i in range(q+1,self.n_qubits):
                            t2.append(i,Operator('I'))

                        for i in range(s):
                            t3.append(i,Operator('z'))
                        t3.append(s,Operator('-'))
                        for i in range(s+1,self.n_qubits):
                            t3.append(i,Operator('I'))

                        for i in range(r):
                            t4.append(i,Operator('z'))
                        t4.append(r,Operator('-'))
                        for i in range(r+1,self.n_qubits):
                            t4.append(i,Operator('I'))

                        
                        res = t1*t2
                        
                        
                        res = res.defactor()

                        for oplist in res:
                            oplist.calculate_factor()
                        


                        res2 = []
                        for oplist in res:
                            res2.append(oplist*t3)
                        res = res2
                        
                        
                        res2 = []
                        for oplist in res:
                            res2.extend(oplist.defactor())
                            
                           
                        res = res2
                        res2 = []
                        for oplist in res:
                            res2.append(oplist*t4)

                        for oplist in res:
                            oplist.calculate_factor()
                        
                        res = res2
                        res2 = []
                        for oplist in res:
                            res2.extend(oplist.defactor())

                        if remove_identity:
                            res2 = self.remove_identity(res2)
                        
                        

                        res = res2
                        res2 = []
                        for oplist in res:
                            new_list = OperatorList(n_qubits)
                            for i in range(n_qubits):
                                new_list.append(i,oplist.get(i).ladder2pauli())
                            res2.append(new_list)
                        res = res2
                        res2 = []
                        for oplist in res:
                            res2.extend(oplist.defactor())

                        if remove_identity:
                            res2 = self.remove_identity(res2)

                        for oplist in res2:
                            oplist.calculate_factor()

                        res = res2
                        res2 = []
                        for oplist in res:
                            oplist.factor *= 4*h_pqrs[p,q,r,s]
                            res2.append(oplist)


                        result.extend(res2)
        
        result = self.add_equals(result)
        return(result)









class Operator:
    def __init__(self,operation):
        self.op = operation.lower()
        if operation == 'I':
            self.op = ''
        self.factor = 1
        self.im = False
        self.ladder = ['+','-']
        self.pauli= ['x','y','z']
        if self.op in self.pauli:
            self.ind = self.pauli.index(self.op)
        if self.op in self.ladder:
            self.ind = self.ladder.index(self.op)

    def __mul__(self,other):
        getSingle = self.getSingle
        getDouble = self.getDouble
        mulcopy = self.mulcopy
        # PAULI
        if self.op in self.pauli:
            if other.op == self.op:
                i,factor = mulcopy(other)
                self.op = ''
                self.factor = factor
                self.im = i
                return self
            elif other.op != self.op and other.op in self.pauli:
                op_ind = 3 - (self.ind + other.ind)
                i,factor = mulcopy(other)
                self.op = self.pauli[op_ind]
                self.check_ind()
                self.im = i
                self.factor = factor
                self.im = not self.im
                if 3*self.ind + other.ind not in [2,3,7]:
                    self.factor *= -1
                return self
            # LADDER
            elif other.op in self.ladder:
                if self.op == 'x':
                    z_fact = eval('{}1'.format(other.op))
                    return getDouble(other,'I','z',0.5,z_fact)
                if self.op == 'y':
                    z_fact = eval('{}1'.format(other.op))
                    return getDouble(other,'I','z',0.5,z_fact,imag=True) 
                if self.op == 'z':
                    fact = eval('{}1'.format(other.op))
                    return getSingle(other,other.op,fact)
            else: # If identity
                self.op += other.op
                self.check_ind()
                i,factor = mulcopy(other)
                self.im = i
                self.factor = factor
                return self
        # LADDER
        elif self.op in self.ladder:
            # LADDER
            if other.op in self.ladder:
                if self.ind == other.ind:
                    return 0
                else: # != -> 1/2*(I +- Z)
                    """
                    last operator sign determines + or - for Z
                    """
                    z_fact = eval('{}1'.format(self.op))
                    return getDouble(other,'I','z',0.5,z_fact)
            # PAULI
            elif other.op in self.pauli:
                if other.op == 'x':
                    z_fact = eval('{}1'.format(other.ladder[(self.ind+1)%2]))
                    return getDouble(other,'I','z',0.5,z_fact)
                elif other.op == 'y':
                    z_fact = eval('{}1'.format(other.ladder[(self.ind+1)%2]))
                    return getDouble(other,'I','z',0.5,z_fact,imag=True) 
                elif other.op == 'z':
                    fact = eval('{}1'.format(self.ladder[(self.ind+1)%2]))
                    return getSingle(other,self.op,fact)
            else: #If identity
                self.op += other.op
                self.check_ind()
                i,factor = mulcopy(other)
                self.im = i
                self.factor = factor
                return self

        # OTHER/IDENTITY
        else:
            self.op += other.op
            self.check_ind()
            i,factor = mulcopy(other)
            self.im = i
            self.factor = factor
            return self

    def __str__(self):
        sign_fact = ''
        if self.factor < 0:
            if self.factor == -1:
                sign_fact = '-'
            else:
                sign_fact = str(self.factor)
        else:
            if self.factor != 1:
                sign_fact = str(self.factor)

        return '{}{}{}{}'.format(sign_fact,
                                 'i' if self.im else '',
                                 '*' if sign_fact != '' else '',
                                 self.op if self.op != '' else 'I')
    
    def __invert__(self): # Assume unitary
        # LADDER
        if self.op in self.ladder:
            self.ind = (self.ind + 1)%2
            self.op = self.ladder[self.ind]
            if self.im:
                self.factor *= -1
            return self.op
        # OTHER/ IDENTITY PAULI
        else:
            if self.im:
                self.factor *= -1
            return self

    def check_ind(self):
        if self.op in self.pauli:
            self.ind = self.pauli.index(self.op)
        if self.op in self.ladder:
            self.ind = self.ladder.index(self.op)

    def getSingle(self,other,new,sign=1):
        """
        Handle factor and imaginary part of multiplication 
        of two operators when single operator is the product.
        """
        op1 = Operator(new)
        i,factor = self.mulcopy(other)
        op1.factor = factor*sign
        op1.im = i
        return op1

    def getDouble(self,other,new1,new2,factor,sign,imag=False):
        """
        When multiplication of two operators lead to summation of two new operators
        new1,new2 -> str of new operators
        factor    -> Factor of sum
        sign      -> Sign between operators
        """
        op1 = Operator(new1)
        op2 = Operator(new2)
        i,new_factor = self.mulcopy(other)
        op1.factor *= factor*new_factor
        op2.factor *= factor*new_factor*sign
        if imag:
            if i:
                op1.factor *= -1
                op2.factor *= -1
            else:
                op1.im = True
                op2.im = True
        else:
            op1.im = i
            op2.im = i
        return [op1,op2]

    def mulcopy(self,other):
        """
        When multiplying two operators. Will handle factor and imaginary part.
        """
        i1 = self.im
        i2 = other.im
        i = False
        factor = self.factor*other.factor
        if i1 and not i2 or i2 and not i1:
            i = True
        if i1 and i2:
            i = False
            factor *= -1
        if not i1 and not i2:
            i = False
        return i,factor
    
    def ladder2pauli(self):
        if self.op not in self.ladder:
            #print('Operator is not ladder')
            return self
        else:
            op1 = Operator('x')
            op2 = Operator('y')
            op1.im = self.im
            if self.im:
                op2.factor *= -1
                op2.im = False
            else:
                op2.im = True
            op1.factor *= self.factor*0.5
            op2.factor *= self.factor*0.5
            if self.op == '-':
                op2.factor *= -1
            return [op1,op2]