<template>
<v-app >
    <v-app-bar app color="dark" dark class="text-center">
        <h2 class="text-center text-uppercase col-12">
            Rețea neurală competitiv-generativă
        </h2>
    </v-app-bar>
    <v-content>
      <v-container>
        <v-row justify="center">
          <v-col cols="12">
          </v-col>
        </v-row>
      </v-container>
        <v-container>
            <v-row justify="center" gradient="to top right, rgba(100,115,201,.33), rgba(25,32,72,.7)"> 
                <Canvas/>
                <Controll/>
            </v-row>
      </v-container>
    </v-content>
    <v-footer color="dark-grey" dark>
        <a class="nav-link white--text" href="https://github.com/kwyz/ganmasterat" target="_blank" style="text-decoration:none;">
            <i class="fab fa-github text-white"></i>  
            <span class="ml-2">Proiectul pe pagina de github</span>
        </a>
        <v-spacer>
        </v-spacer>
        <div>
            Muștuc Vasile &copy; {{ new Date().getFullYear() }}
        </div>
    </v-footer>
  </v-app>
</template>

<script>

export default {
  name: 'App',
  components: {
    Canvas: () => import("@/components/Canvas/Index.vue"),
    Controll: () => import("@/components/Controll/Index.vue"),
  },

  data: () => ({
    //
  }),
};
</script>
