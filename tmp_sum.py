#coding:utf-8
import sys
import traceback
import json
from HTMLParser import HTMLParser
reload(sys)
sys.setdefaultencoding('utf-8')

title = "bdbkLemmaTitle"
url = "bdbkLemmaUrl"
cate = "bdbkPsCategory"
kv_data= "bdbkKvData"
content= "bdbkContent"
tag = "bdbklemmaTags"
migrate = "bdbkMigration"
delete = "bdbkDelete"
syn = "bdbkSynLemma"
id = "ID"
pv = "bdbkPV"
view_id = "VIEW_ID"
sum_content = "bdbkSummaryContent" #è¯ææ€»ç
content_con="content"
no_str = "NULL".strip()
pv_t="totalPV"
ldesc="bdbkLemmaDesc"
h_list=["h2", "h3", "h4"]

class MyHtmlParser(HTMLParser):
    def __init__(self):   
        HTMLParser.__init__(self)  
        self.data=[]
        self.now_tag=""
        self.end_tag=""
        self.flag=0
    def handle_starttag(self, tag, attrs):
        """
        0:h
        1:data
        2:del
        """
#self.data.append((tag, 0))
#self.s_tag = tag
        if tag =='a':
            self.data.append([tag, 0])
        if tag == 'h2':
            self.data.append([tag, 0])
        if tag =='h3':
            self.data.append([tag, 0])
        if tag == 'h4':
            self.data.append([tag, 0])
        if tag =='h5':
            self.data.append([tag, 0])
        if tag =='h6':
            self._tag.append([tag, 0])
        if tag =='p':
            self.data.append([tag, 0])
            self.flag = 1
        if tag == 'b':
            self.data.append([tag, 0])
            self.flag = 2
        if tag == 'table' or tag.strip() == 'td' or tag.strip() == 'tr' or tag.strip() == 'th':
            self.flag = 3
        if tag == 'iframe':
            for (variable, value) in attrs:
                if variable == 'data-type':
                    if value != 'album' and value != 'map' and value != 'video':
                        self.flag = 1
                        self.data.append([tag, 1])
                        break
        self.now_tag = tag
        #print 'debug_tag_start: ', tag
        
    def handle_endtag(self ,tag):
        self.flag = 0
        self.end_tag=tag.rstrip()
        #print 'debug_tag_end: ', tag
        #if tag=="table":
         #   self.flag=True
        
        """
        if tag == 'h2':
           self._tag.append(tag)
        if tag =='h3':
           self._tag.append(tag)
        if tag == 'h4':
           self._tag.append(tag)
        if tag =='h5':
           self._tag.append(tag)
        if tag =='h6':
           self._tag.append(tag)
        if tag == 'p':
           self._tag.append(tag)
       """
            
    def handle_data(self, data):
        """
        if self.now_tag=="table":
            index =0
            for each in self.data[::-1]:
                item1=each[0]
                item2=each[1]
                if item2==0:
                    if item1=="h2":
                        self.data=self.data[:len(self.data)-index]
                        break
                index+=1
        """
        #print 'debug', data, self.flag, self.now_tag
        if (self.now_tag!='sup'):
            #if self.now_tag=='p' or self.end_tag=='br'or self.now_tag in h_list:
            data = data.strip('\n').strip('\r')
            
            if self.flag == 2:
                data="<b>"+data
                self.flag = 0
            if self.flag == 1:
                data="\x06" + data
                self.flag = 0
            if self.flag == 3:
                data="<table>"+data
                self.flag = 0
            data_index = 0
            
            if self.end_tag=="table" or self.end_tag=="tr" or self.end_tag=="td" or self.end_tag=="th":
                for index in xrange(len(self.data)-1, 0, -1):
                    if self.data[index][1] == 1:
                        data_index = index
                        break
                if data_index >= 0 and data_index < len(self.data):
                    self.data[data_index][0]+="</table>"
                self.end_tag = ''
            
            data_index = 0
            if self.end_tag=="b":
                for index in xrange(len(self.data)-1, 0, -1):
                    if self.data[index][1] == 1:
                        data_index = index
                        break
                if data_index >= 0 and data_index < len(self.data):
                    self.data[data_index][0]+="</b>"
                self.end_tag = ''
            
            data_index = 0
            if self.end_tag=="p":
                for index in xrange(len(self.data)-1, 0, -1):
                    if self.data[index][1] == 1:
                        data_index = index
                        break
                if data_index >= 0 and data_index < len(self.data):
                    self.data[data_index][0]+="\x06"
                self.end_tag = ''
                
            #print "debug " + data
            self.data.append([data, 1])
        """
        if self.s_tag =='h2':
           self.title.append(data)
        if self.s_tag =='h3':
           self.title.append(data)
        if self.s_tag =='h4':
           self.title.append(data)
        if self.s_tag =='h5':
           self.title.append(data)
        if self.s_tag =='h6':
           self.title.append(data)
        if self.s_tag =='p':
        if self.s_tag =='a':
           self.data[]
           self.data.append(data)
    """
        
#print data
#parser = MyHtmlParser() 
def find_data(id, all_data):
    len_data = len(all_data)
    for i in xrange(id, len_data):
        tag = all_data[i][1]
        if tag == 1:
           return i
    return 0
sp1 ='\x01'
sp2 = ''
sp3='\x02'
h_list=["h1", "h2", "h3", "h4", "h5", "h6"]

for line in sys.stdin:
  try:
    line = json.loads(line)
    parser = MyHtmlParser()
    q_title= line[title] if title in line else no_str
    q_url = line[url] if url in line else no_str
    q_cate = line[cate] if cate in line else no_str 
    q_tag = line[tag] if tag in line else no_str
    q_pv_t = line[pv][pv_t] if pv in line else no_str
    q_content = line[content][content_con] if content in line else no_str
    q_syn = line[syn] if syn in line else no_str
    #q_content_summary = line[summary_content][content_con] if summary_content in line else no_str
    q_content = q_content.replace('\\x0d', '\x0d').replace('\\x0a', '\x0a').replace('\\r\\n', '\x03').replace('\\n', '\x05')
    q_ldesc= line[ldesc] if ldesc in line else no_str
    q_sum_content= line[sum_content]["content"] if sum_content in line else no_str
    is_migrate = True if migrate in line else False
    is_delete = True if delete in line else False
    #if q_content=='NULL'or q_content=='null'or q_content=="":
     #  continue
    #if q_sum_content!="NULL" and q_sum_content!='null'and q_sum_content!="" and not q_sum_content is None:
     #   parser_sum = MyHtmlParser()
      #  parser_sum.feed(q_sum_content)
       # sum_data = parser_sum.data
        #sum_data_list=[]
        #for each in sum_data:
         #   tag = each[1]
          #  if tag==1:
           #     sum_data_list.append(each[0])
    q_cate_list =[]
    if type(q_cate)==list: 
        q_cate_list.extend(q_cate)
    else:
        q_cate_list.append(q_cate)
    if type(q_tag)==list:
        q_cate_list.extend(q_tag)
    else:
        q_cate_list.append(q_tag)
    q_syn_list = []
    if type(q_syn)==list:
        for item in q_syn:
            if "lemmaTitle" in item:
                q_syn_list.append(item["lemmaTitle"])
    else:
        if "lemmaTitle" in q_syn:
            q_syn_list.append(q_syn["lemmaTitle"])
        #print q_title, '\t', no_str, '\t', " ".join(sum_data_list), '\t', q_url, '\t', q_pv_t
    q_content = q_content.replace('<br>', '\x06').replace('<br/>', '\x06').replace('<br />', '\x06')
    parser.feed(q_content)
    #print len(parser._tag), ' ', len(parser._data)
    d_out=[]
    out_title =""
    all_data = parser.data
    len_data = len(all_data)
    i = 0
    title_list={}
    h_index=0
    sub_index=0
    while i < len_data:
        data = all_data[i][0]
        tag = all_data[i][1]
        if tag == 0:
           if  data =='h2':
               if out_title=="":
                   out_title =no_str
                   #print q_title,'\t',out_title, "\t", sp2.join(d_out), '\t', q_url, '\t', q_pv_t
                   if len(d_out) !=0:
                       print q_title,'\t',out_title, "\t", sub_index, '\t', sp2.join(d_out), '\t', q_url+"#"+str(h_index), '\t', q_pv_t, '\t', sp1.join(q_cate_list), '\t', q_ldesc, '\t', q_sum_content, '\t', str(is_migrate), '\t', str(is_delete), '\t', sp1.join(q_syn_list)
               else:
                   print q_title,'\t',out_title, "\t", sub_index, '\t', sp2.join(d_out), '\t', q_url+"#"+str(h_index), '\t', q_pv_t, '\t', sp1.join(q_cate_list), '\t',q_ldesc, '\t', q_sum_content, '\t', str(is_migrate), '\t', str(is_delete), '\t', sp1.join(q_syn_list)
               id = find_data(i+1, all_data)
               if id ==0:
                  print >>sys.stderr, 'error2', q_title
                  break
               i = id
               data = all_data[i][0]
               out_title = data
               title_list={}
               title_list[0]= out_title
               d_out=[]
               h_index+=1
               sub_index=0
           elif  data =='h3':
               if out_title=="":
                   out_title = no_str
               print q_title, '\t',out_title, "\t", sub_index, '\t', sp2.join(d_out), '\t', q_url+"#"+str(h_index), '\t', q_pv_t, '\t', sp1.join(q_cate_list), '\t',q_ldesc, '\t', q_sum_content, '\t', str(is_migrate), '\t', str(is_delete), '\t', sp1.join(q_syn_list)
               id = find_data(i+1, all_data)
               if id == 0:
                  print >>sys.stderr, 'error3'
                  break
               i = id
               data = all_data[i][0]
               if 0 not in title_list:
                  out_title = data
               else:
                  out_title = title_list[0]+sp1+data
               title_list[1]= out_title
               d_out=[]
               sub_index+=1
        elif tag == 1:
             last_d = all_data[i-1][0]
             last_t = all_data[i-1][1]
             if last_d=='a'and last_t==0:
               if len(d_out)==0:
                  d_out.append(data)
               else:
                  if (i-2)>=0:
                     tag = all_data[i-2][1]
                     if tag == 1:
                        d_out[-1]+=data
                     else:
                        d_out.append(data)
               if i+1<len_data:
                  i= i+1
                  tag = all_data[i][1]
                  data = all_data[i][0]
                  if tag == 1:
                     d_out[-1]+=data
                  else:
                     i -=1
             else:
               d_out.append(data)
        i+=1
    if out_title=="":
        out_title=no_str
    print q_title, '\t',out_title, "\t", sub_index, '\t', sp2.join(d_out), '\t', q_url+"#"+str(h_index), '\t', q_pv_t, '\t', sp1.join(q_cate_list), '\t',q_ldesc, '\t', q_sum_content, '\t', str(is_migrate), '\t', str(is_delete), '\t', sp1.join(q_syn_list)
  except Exception, e:
      print>>sys.stderr, traceback.print_exc()
      #print >>sys.stderr,q_content
      continue
